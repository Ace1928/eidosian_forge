from eidosian_core import eidosian
"""Conversation Manager Module.

This module provides the conversation management functionality for Word Forge,
enabling multi-turn conversations with language model integration. It handles
conversation persistence, message threading, and orchestrates multiple model
types for generating contextually aware responses.

Key Components:
    ConversationManager: Orchestrates conversation sessions and message flow
    ModelContext: Tracks conversation state and context between turns
    ConversationError: Base exception for conversation-related errors

The manager integrates with:
    - Database for conversation and message persistence
    - Emotion analysis for sentiment-aware responses
    - Vector store for semantic similarity search
    - Graph manager for relationship context
    - Multiple language models (lightweight, reflexive, affective, identity)

Architecture:
    User Input → Context Assembly → Model Selection → Response Generation →
    Emotion Analysis → Message Storage → Context Update

Example:
    >>> conv_manager = ConversationManager(db_manager, emotion_manager)
    >>> result = conv_manager.start_conversation()
    >>> if result.is_success:
    ...     conv_id = result.unwrap()
    ...     msg_id = conv_manager.add_message(conv_id, "user", "Hello!")
"""

import sqlite3
import traceback
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, cast

# Import Result and Error types correctly
from word_forge.configs.config_essentials import (
    Error,
    ErrorCategory,
    ErrorSeverity,
    Result,
)

# Import protocols and types from conversation_types
from word_forge.conversation.conversation_types import (
    AffectiveLexicalModel,
    ConversationDict,
    IdentityModel,
    LightweightModel,
    MessageDict,
    ModelContext,
    ReflexiveModel,
)
from word_forge.database.database_manager import DatabaseError, DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.vectorizer.vector_store import VectorStore
from word_forge.queue.queue_manager import QueueManager
from word_forge.parser.parser_refiner import TermExtractor


# --- Custom Exceptions ---
class ConversationError(Exception):
    """
    Base exception for errors specific to conversation management operations.

    Inherits from Exception, providing a specific type for catching
    conversation-related issues distinct from general database errors.

    Attributes:
        message (str): A description of the error.
        original_exception (Optional[Exception]): The underlying exception, if any.
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Initializes a ConversationError.

        Args:
            message: The error message describing the issue.
            original_exception: The optional original exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Returns the string representation of the error."""
        if self.original_exception:
            return f"{self.message}: {self.original_exception}"
        return self.message


class ConversationNotFoundError(ConversationError):
    """
    Raised specifically when a conversation cannot be found in the database.

    Inherits from ConversationError for categorization.
    """

    def __init__(self, conversation_id: int):
        """
        Initializes a ConversationNotFoundError.

        Args:
            conversation_id: The ID of the conversation that was not found.
        """
        super().__init__(f"Conversation with ID {conversation_id} not found.")
        self.conversation_id = conversation_id


# --- SQL Query Constants ---
SQL_CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT DEFAULT 'ACTIVE' NOT NULL,
    created_at REAL DEFAULT (strftime('%s','now')) NOT NULL,
    updated_at REAL DEFAULT (strftime('%s','now')) NOT NULL
);
"""

SQL_CREATE_CONVERSATION_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp REAL DEFAULT (strftime('%s','now')) NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""

SQL_START_CONVERSATION = """
INSERT INTO conversations (status) VALUES ('ACTIVE');
"""

SQL_END_CONVERSATION = """
UPDATE conversations
SET status = 'COMPLETED',
    updated_at = strftime('%s','now')
WHERE id = ?;
"""

SQL_ADD_MESSAGE = """
INSERT INTO conversation_messages (conversation_id, speaker, text)
VALUES (?, ?, ?);
"""

SQL_UPDATE_CONVERSATION_TIMESTAMP = """
UPDATE conversations
SET updated_at = strftime('%s','now')
WHERE id = ?;
"""

SQL_GET_CONVERSATION = """
SELECT id, status, created_at, updated_at
FROM conversations
WHERE id = ?;
"""

SQL_GET_MESSAGES = """
SELECT
    cm.id,
    cm.speaker,
    cm.text,
    cm.timestamp
FROM conversation_messages cm
WHERE cm.conversation_id = ?
ORDER BY cm.timestamp ASC;
"""


class ConversationManager:
    """
    Manages conversation sessions, messages, and multi-model response generation.

    Orchestrates interactions with the database and a sequence of language models
    (Reflexive, Lightweight, Affective/Lexical, Identity) to manage chat flows.
    Provides an interface adhering to principles of precision, clarity,
    and robust error handling using the Result pattern.

    Attributes:
        db_manager (DBManager): Instance for database interactions.
        emotion_manager (EmotionManager): Instance for emotion analysis.
        graph_manager (GraphManager): Instance for graph database access.
        vector_store (VectorStore): Instance for vector similarity searches.
        reflexive_model (ReflexiveModel): Protocol implementation for initial response.
        lightweight_model (LightweightModel): Protocol implementation for routing/basic processing.
        affective_model (AffectiveLexicalModel): Protocol implementation for core response generation.
        identity_model (IdentityModel): Protocol implementation for final refinement.
    """

    def __init__(
        self,
        db_manager: DBManager,
        emotion_manager: EmotionManager,
        graph_manager: GraphManager,
        vector_store: VectorStore,
        reflexive_model: ReflexiveModel,
        lightweight_model: LightweightModel,
        affective_model: AffectiveLexicalModel,
        identity_model: IdentityModel,
        queue_manager: Optional[QueueManager[str]] = None,
        term_extractor: Optional[TermExtractor] = None,
    ) -> None:
        """
        Initializes the ConversationManager with dependencies and models.

        Ensures necessary database tables are created upon instantiation.

        Args:
            db_manager: Database manager instance for persistence.
            emotion_manager: Emotion manager instance for analysis.
            graph_manager: Graph manager instance for knowledge access.
            vector_store: Vector store instance for similarity searches.
            reflexive_model: The initial reflexive model instance.
            lightweight_model: The routing/basic processing model instance.
            affective_model: The core understanding and response model instance.
            identity_model: The personality and refinement model instance.
            queue_manager: Optional queue for discovered terms.
            term_extractor: Custom term extractor instance.

        Raises:
            ConversationError: If initialization fails, particularly during table creation.
        """
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self.graph_manager = graph_manager
        self.vector_store = vector_store
        self.reflexive_model = reflexive_model
        self.lightweight_model = lightweight_model
        self.affective_model = affective_model
        self.identity_model = identity_model
        self.queue_manager = queue_manager
        self.term_extractor = term_extractor or TermExtractor()
        try:
            self._create_tables()
        except ConversationError as e:
            print(f"Error during ConversationManager initialization: {e}")
            raise

    @contextmanager
    def _db_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Provides a managed database connection context using the DBManager.

        Ensures the connection uses `sqlite3.Row` for dictionary-like row access
        and handles potential `DatabaseError` exceptions from the manager,
        wrapping them in `ConversationError`.

        Yields:
            sqlite3.Connection: An active SQLite database connection configured with Row factory.

        Raises:
            ConversationError: If obtaining or managing the connection fails.
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            with self.db_manager.get_connection() as db_conn:
                conn = db_conn
                conn.row_factory = sqlite3.Row
                yield conn
        except DatabaseError as e:
            raise ConversationError(
                f"Failed to get database connection via DBManager: {e}",
                original_exception=e,
            ) from e
        except Exception as e:
            raise ConversationError(
                f"Unexpected error during database connection context: {e}",
                original_exception=e,
            ) from e

    def _create_tables(self) -> None:
        """
        Ensures necessary database tables exist for conversations and messages.

        Executes `CREATE TABLE IF NOT EXISTS` statements within a transaction.

        Raises:
            ConversationError: If executing the SQL statements fails.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_CREATE_CONVERSATIONS_TABLE)
                cursor.execute(SQL_CREATE_CONVERSATION_MESSAGES_TABLE)
                conn.commit()
        except sqlite3.Error as e:
            raise ConversationError(
                f"SQLite error initializing conversation tables: {e}",
                original_exception=e,
            ) from e
        except ConversationError as e:
            raise e
        except Exception as e:
            raise ConversationError(
                f"Unexpected error initializing conversation tables: {e}",
                original_exception=e,
            ) from e

    def _queue_terms_from_text(self, text: str) -> None:
        """Extract words from ``text`` and enqueue unseen terms for processing."""

        if not self.queue_manager:
            return

        priority, standard = self.term_extractor.extract_terms(text, [], "")
        for term in priority + standard:
            if not self.db_manager.word_exists(term):
                self.queue_manager.enqueue(term)

    @eidosian()
    def start_conversation(self) -> Result[int]:
        """
        Initiates a new conversation record in the database.

        Sets the initial status to 'ACTIVE' and records the creation timestamp.
        Uses the Result pattern for explicit success/failure signaling.

        Returns:
            Result[int]: On success, contains the unique ID of the newly created
                         conversation. On failure, contains an Error object.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_START_CONVERSATION)
                conversation_id = cursor.lastrowid
                conn.commit()
                if conversation_id is None:
                    error = Error.create(
                        message="Failed to retrieve conversation ID after insertion (lastrowid was None).",
                        code="DB_INSERT_NO_ID",
                        category=ErrorCategory.UNEXPECTED,
                        severity=ErrorSeverity.ERROR,
                    )
                    return Result[int].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )
                print(f"Started new conversation with ID: {conversation_id}")
                return Result[int].success(conversation_id)
        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error starting new conversation: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={"sql_operation": "start_conversation"},
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error starting conversation: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={"sql_operation": "start_conversation"},
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error starting conversation: {e}",
                code="UNEXPECTED_START_CONV_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "start_conversation",
                    "exception_type": type(e).__name__,
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    @eidosian()
    def end_conversation(self, conversation_id: int) -> Result[None]:
        """
        Marks an existing conversation as 'COMPLETED'.

        Updates the conversation's status and `updated_at` timestamp.
        Uses the Result pattern for explicit success/failure signaling.

        Args:
            conversation_id: The ID of the conversation to mark as completed.

        Returns:
            Result[None]: Contains None on success, or an Error object on failure.
                          Specifically returns a CONVERSATION_NOT_FOUND error
                          if the ID does not exist.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_END_CONVERSATION, (conversation_id,))
                conn.commit()
                if cursor.rowcount == 0:
                    error = Error.create(
                        message=f"Conversation with ID {conversation_id} not found to end.",
                        code="CONVERSATION_NOT_FOUND",
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.WARNING,
                        context={"conversation_id": str(conversation_id)},
                    )
                    return Result[None].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )
                print(f"Ended conversation with ID: {conversation_id}")
                return Result[None].success(None)
        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error ending conversation {conversation_id}: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "end_conversation",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[None].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error ending conversation: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "end_conversation",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[None].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error ending conversation {conversation_id}: {e}",
                code="UNEXPECTED_END_CONV_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "end_conversation",
                    "conversation_id": str(conversation_id),
                    "exception_type": type(e).__name__,
                },
            )
            return Result[None].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    @eidosian()
    def add_message(
        self,
        conversation_id: int,
        speaker: str,
        text: str,
        generate_response: bool = False,
    ) -> Result[int]:
        """
        Adds a message and optionally triggers multi-model response generation.

        Persists the message, updates conversation timestamp, processes emotion,
        and if `generate_response` is True and speaker is not 'Assistant',
        initiates the response pipeline. Uses the Result pattern.

        Args:
            conversation_id: The ID of the target conversation.
            speaker: Identifier of the message sender (e.g., "User", "Assistant").
                     Case-insensitive check for "Assistant" to prevent self-reply loops.
            text: The textual content of the message. Cannot be empty or whitespace.
            generate_response: If True and speaker is not "Assistant", triggers
                               response generation. Defaults to False.

        Returns:
            Result[int]: On success, contains the unique ID of the newly added
                         *input* message. On failure, contains an Error object.
                         If response generation fails, the input message is still
                         added, but the overall result reflects the generation failure.

        Raises:
            ValueError: If the provided message text is empty or purely whitespace.
                        (This is a programming error, hence direct raise).
        """
        if not text or not text.strip():
            raise ValueError("Message text cannot be empty or whitespace.")

        message_id: Optional[int] = None
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(SQL_ADD_MESSAGE, (conversation_id, speaker, text))
                message_id = cursor.lastrowid

                cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (conversation_id,))
                conn.commit()

                if message_id is None:
                    error = Error.create(
                        message="Failed to retrieve message ID after insertion.",
                        code="DB_INSERT_NO_ID",
                        category=ErrorCategory.UNEXPECTED,
                        severity=ErrorSeverity.ERROR,
                        context={
                            "conversation_id": str(conversation_id),
                            "speaker": speaker,
                        },
                    )
                    return Result[int].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )

                print(
                    f"Added message {message_id} from {speaker} to conversation {conversation_id}"
                )

                try:
                    self.emotion_manager.process_message(message_id, text)
                except Exception as emotion_e:
                    print(
                        f"Warning: Failed to process emotion for message {message_id} "
                        f"in conversation {conversation_id}: {emotion_e}"
                    )

                if speaker.lower() != "assistant":
                    self._queue_terms_from_text(text)

            if generate_response and speaker.lower() != "assistant":
                print(
                    f"Triggering response generation for conversation {conversation_id}..."
                )
                response_result: Result[int] = self.generate_and_add_response(
                    conversation_id, text, speaker
                )
                if response_result.is_failure:
                    error_details = "Unknown error"
                    error_code = "RESPONSE_GENERATION_FAILED"
                    if response_result.error:
                        error_details = response_result.error.message
                        error_code = response_result.error.code
                    print(
                        f"Error generating response for conversation {conversation_id}: "
                        f"Code: {error_code}, Details: {error_details}"
                    )
                    failure_error = response_result.error or Error.create(
                        code="RESPONSE_GEN_UNKNOWN_FAILURE",
                        message="Response generation failed without specific error.",
                        context={
                            "conversation_id": str(conversation_id),
                            "speaker": speaker,
                            "message_id": str(message_id),
                            "error_details": error_details,
                            "error_code": error_code,
                        },
                        category=ErrorCategory.UNEXPECTED,
                        severity=ErrorSeverity.ERROR,
                    )
                    return Result[int].failure(
                        failure_error.code,
                        failure_error.message,
                        failure_error.context,
                        failure_error.category,
                        failure_error.severity,
                    )
                else:
                    print(
                        f"Successfully generated and added response (ID: {response_result.unwrap()})"
                    )
                    return Result[int].success(message_id)
            else:
                return Result[int].success(message_id)

        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error adding message to conversation {conversation_id}: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "add_message",
                    "conversation_id": str(conversation_id),
                    "speaker": speaker,
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error adding message: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "add_message",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error adding message or generating response for conversation {conversation_id}: {e}",
                code="UNEXPECTED_ADD_MSG_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "add_message",
                    "conversation_id": str(conversation_id),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    @eidosian()
    def generate_and_add_response(
        self, conversation_id: int, last_user_text: str, last_user_speaker: str
    ) -> Result[int]:
        """
        Generates and adds an assistant response via the multi-model pipeline.

        Orchestrates: Context Prep -> Reflexive -> Lightweight -> Affective -> Identity -> Add Response.
        Uses the Result pattern throughout for robust error handling.

        Args:
            conversation_id: ID of the current conversation.
            last_user_text: Text of the last message triggering the response.
            last_user_speaker: Speaker of the last message triggering the response.

        Returns:
            Result[int]: Contains the ID of the added assistant message on success,
                         or an Error object detailing the failure.
        """
        print(f"[{conversation_id}] Preparing context for response generation...")
        conv_result = self.get_conversation(conversation_id)
        if conv_result.is_failure:
            failure_error = conv_result.error or Error.create(
                message="Failed to get conversation data before response generation.",
                code="GET_CONV_FAILED_PRE_RESPONSE",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
            )
            return Result[int].failure(
                failure_error.code,
                failure_error.message,
                failure_error.context,
                failure_error.category,
                failure_error.severity,
            )
        conversation_data = conv_result.unwrap()

        history_limit = 20
        limited_history = conversation_data["messages"][-history_limit:]

        context: ModelContext = {
            "conversation_id": conversation_id,
            "history": limited_history,
            "current_input": last_user_text,
            "speaker": last_user_speaker,
            "db_manager": self.db_manager,
            "emotion_manager": self.emotion_manager,
            "graph_manager": self.graph_manager,
            "vector_store": self.vector_store,
            "reflexive_output": None,
            "intermediate_response": None,
            "affective_state": None,
            "identity_state": None,
            "additional_data": {},
        }
        print(
            f"[{conversation_id}] Context prepared with {len(limited_history)} messages."
        )

        try:
            print(f"[{conversation_id}] Calling Reflexive Model...")
            reflexive_result = self.reflexive_model.generate_reflex(context)
            if reflexive_result.is_failure:
                reflex_error_msg = "Unknown reflexive failure"
                reflex_error_code = "REFLEX_UNKNOWN"
                reflex_error_context: Optional[Dict[str, str]] = None
                if reflexive_result.error:
                    reflex_error_msg = reflexive_result.error.message
                    reflex_error_code = reflexive_result.error.code
                    reflex_error_context = reflexive_result.error.context

                print(
                    f"[{conversation_id}] Warning: Reflexive model failed: {reflex_error_msg}. Proceeding."
                )
                context["additional_data"]["reflexive_error"] = {
                    "message": reflex_error_msg,
                    "code": reflex_error_code,
                    "context": reflex_error_context or {},
                }

            print(f"[{conversation_id}] Calling Lightweight Model...")
            lightweight_result = self.lightweight_model.process(context)
            if lightweight_result.is_failure:
                print(f"[{conversation_id}] Error: Lightweight model failed.")
                failure_error = lightweight_result.error or Error.create(
                    message="Lightweight model failed without specific error.",
                    code="LIGHTWEIGHT_UNKNOWN_FAILURE",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                )
                return Result[int].failure(
                    failure_error.code,
                    failure_error.message,
                    failure_error.context,
                    failure_error.category,
                    failure_error.severity,
                )
            context = lightweight_result.unwrap()
            print(f"[{conversation_id}] Lightweight Model finished.")

            print(f"[{conversation_id}] Calling Affective/Lexical Model...")
            affective_result = self.affective_model.generate_core_response(context)
            if affective_result.is_failure:
                print(f"[{conversation_id}] Error: Affective model failed.")
                failure_error = affective_result.error or Error.create(
                    message="Affective model failed without specific error.",
                    code="AFFECTIVE_UNKNOWN_FAILURE",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                )
                return Result[int].failure(
                    failure_error.code,
                    failure_error.message,
                    failure_error.context,
                    failure_error.category,
                    failure_error.severity,
                )
            context = affective_result.unwrap()
            intermediate_resp = context.get("intermediate_response", "")
            if intermediate_resp:
                preview = intermediate_resp[:50]
            else:
                preview = "<empty response>"
            print(f"[{conversation_id}] Affective Model generated: '{preview}...'")

            print(f"[{conversation_id}] Calling Identity Model...")
            identity_result = self.identity_model.refine_response(context)
            if identity_result.is_failure:
                print(f"[{conversation_id}] Error: Identity model failed.")
                failure_error = identity_result.error or Error.create(
                    message="Identity model failed without specific error.",
                    code="IDENTITY_UNKNOWN_FAILURE",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                )
                return Result[int].failure(
                    failure_error.code,
                    failure_error.message,
                    failure_error.context,
                    failure_error.category,
                    failure_error.severity,
                )
            final_response_text = identity_result.unwrap()
            print(
                f"[{conversation_id}] Identity Model produced final: '{final_response_text[:50]}...'"
            )

            print(f"[{conversation_id}] Adding final assistant response to DB...")
            add_assistant_result = self._add_assistant_message_internal(
                conversation_id, final_response_text
            )
            if add_assistant_result.is_failure:
                print(
                    f"[{conversation_id}] Error: Failed to add assistant message to DB."
                )
                failure_error = add_assistant_result.error or Error.create(
                    message="Failed to add assistant message internally without specific error.",
                    code="ADD_ASSISTANT_INTERNAL_FAILURE",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                )
                return Result[int].failure(
                    failure_error.code,
                    failure_error.message,
                    failure_error.context,
                    failure_error.category,
                    failure_error.severity,
                )

            assistant_message_id = add_assistant_result.unwrap()
            print(
                f"[{conversation_id}] Assistant message {assistant_message_id} added."
            )
            return Result[int].success(assistant_message_id)

        except Exception as e:
            print(
                f"[{conversation_id}] Unexpected error during response generation pipeline: {type(e).__name__} - {e}"
            )
            error = Error.create(
                message=f"Unexpected error in response pipeline: {e}",
                code="RESPONSE_PIPELINE_UNEXPECTED_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "conversation_id": str(conversation_id),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    def _add_assistant_message_internal(
        self, conversation_id: int, text: str
    ) -> Result[int]:
        """
        Internal helper to add an 'Assistant' message using the Result pattern.

        Handles database insertion, timestamp update, and best-effort emotion
        processing for the assistant's message.

        Args:
            conversation_id: The ID of the conversation.
            text: The assistant's message text.

        Returns:
            Result[int]: Contains the ID of the newly added assistant message on success,
                         or an Error object on failure.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_ADD_MESSAGE, (conversation_id, "Assistant", text))
                message_id = cursor.lastrowid
                cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (conversation_id,))
                conn.commit()

                if message_id is None:
                    error = Error.create(
                        message="Failed to retrieve assistant message ID after insertion.",
                        code="DB_INSERT_NO_ID",
                        category=ErrorCategory.UNEXPECTED,
                        severity=ErrorSeverity.ERROR,
                        context={"conversation_id": str(conversation_id)},
                    )
                    return Result[int].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )

                try:
                    self.emotion_manager.process_message(message_id, text)
                except Exception as emotion_e:
                    print(
                        f"Warning: Failed to process emotion for assistant message {message_id} "
                        f"in conversation {conversation_id}: {emotion_e}"
                    )

                return Result[int].success(message_id)

        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error adding assistant message: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "_add_assistant_message_internal",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error adding assistant message: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "_add_assistant_message_internal",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error adding assistant message: {e}",
                code="UNEXPECTED_ADD_ASSISTANT_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "_add_assistant_message_internal",
                    "conversation_id": str(conversation_id),
                    "exception_type": type(e).__name__,
                },
            )
            return Result[int].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    @eidosian()
    def get_conversation(self, conversation_id: int) -> Result[ConversationDict]:
        """
        Retrieves full conversation details, including messages.

        Fetches conversation metadata and all associated messages, ordered chronologically.
        Handles potential database errors and non-existent conversations using the Result pattern.

        Args:
            conversation_id: The ID of the conversation to retrieve.

        Returns:
            Result[ConversationDict]: On success, contains a dictionary representing
                                      the conversation. On failure, contains an Error object,
                                      including CONVERSATION_NOT_FOUND if applicable.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(SQL_GET_CONVERSATION, (conversation_id,))
                row: Optional[sqlite3.Row] = cursor.fetchone()

                if not row:
                    error = Error.create(
                        message=f"Conversation with ID {conversation_id} not found.",
                        code="CONVERSATION_NOT_FOUND",
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.WARNING,
                        context={"conversation_id": str(conversation_id)},
                    )
                    return Result[ConversationDict].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )

                conv_data: ConversationDict = {
                    "id": int(cast(int, row["id"])),
                    "status": str(cast(str, row["status"])),
                    "created_at": float(cast(float, row["created_at"])),
                    "updated_at": float(cast(float, row["updated_at"])),
                    "messages": [],
                }

                cursor.execute(SQL_GET_MESSAGES, (conversation_id,))
                messages_rows: List[sqlite3.Row] = cursor.fetchall()

                for m_row in messages_rows:
                    message: MessageDict = {
                        "id": int(cast(int, m_row["id"])),
                        "speaker": str(cast(str, m_row["speaker"])),
                        "text": str(cast(str, m_row["text"])),
                        "timestamp": float(cast(float, m_row["timestamp"])),
                        "emotion": None,
                    }

                    conv_data["messages"].append(message)

                return Result[ConversationDict].success(conv_data)

        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error retrieving conversation {conversation_id}: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "get_conversation",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[ConversationDict].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error retrieving conversation: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "get_conversation",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[ConversationDict].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error retrieving conversation {conversation_id}: {e}",
                code="UNEXPECTED_GET_CONV_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "get_conversation",
                    "conversation_id": str(conversation_id),
                    "exception_type": type(e).__name__,
                },
            )
            return Result[ConversationDict].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    @eidosian()
    def get_conversation_if_exists(
        self, conversation_id: int
    ) -> Result[Optional[ConversationDict]]:
        """
        Retrieves conversation details only if the conversation exists.

        Wraps `get_conversation` but treats `CONVERSATION_NOT_FOUND` as a
        successful outcome returning `None`, rather than a failure. Other
        errors are still propagated as failures.

        Args:
            conversation_id: The ID of the conversation to attempt retrieval for.

        Returns:
            Result[Optional[ConversationDict]]: On success, contains the conversation
                                                 data if found, or None if not found.
                                                 On failure (e.g., database error),
                                                 contains an Error object.
        """
        result = self.get_conversation(conversation_id)

        if result.is_success:
            return Result[Optional[ConversationDict]].success(result.unwrap())
        elif result.error and result.error.code == "CONVERSATION_NOT_FOUND":
            return Result[Optional[ConversationDict]].success(None)
        else:
            failure_error = result.error or Error.create(
                message="Unknown error in get_conversation_if_exists.",
                code="GET_CONV_IF_EXISTS_UNKNOWN_FAILURE",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
            )
            return Result[Optional[ConversationDict]].failure(
                failure_error.code,
                failure_error.message,
                failure_error.context,
                failure_error.category,
                failure_error.severity,
            )

    @eidosian()
    def list_conversations(self, limit: int = 10) -> Result[List[ConversationDict]]:
        """
        List recent conversations with their message counts.

        Retrieves conversations from the database ordered by most recently updated.
        Uses the Result pattern for explicit success/failure signaling.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            Result[List[ConversationDict]]: On success, contains a list of conversation
                dictionaries. On failure, contains an Error object.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        c.id,
                        c.status,
                        c.created_at,
                        c.updated_at,
                        COUNT(cm.id) as message_count
                    FROM conversations c
                    LEFT JOIN conversation_messages cm ON c.id = cm.conversation_id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                conversations: List[ConversationDict] = []
                for row in rows:
                    conv: ConversationDict = {
                        "id": row["id"],
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "message_count": row["message_count"],
                    }
                    conversations.append(conv)
                return Result[List[ConversationDict]].success(conversations)
        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error listing conversations: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={"sql_operation": "list_conversations"},
            )
            return Result[List[ConversationDict]].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error listing conversations: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={"sql_operation": "list_conversations"},
            )
            return Result[List[ConversationDict]].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error listing conversations: {e}",
                code="UNEXPECTED_LIST_CONV_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "list_conversations",
                    "exception_type": type(e).__name__,
                },
            )
            return Result[List[ConversationDict]].failure(
                error.code, error.message, error.context, error.category, error.severity
            )

    @eidosian()
    def get_messages(
        self, conversation_id: int, limit: int = 50
    ) -> Result[List[MessageDict]]:
        """
        Get messages from a specific conversation.

        Retrieves messages from the database ordered by timestamp.
        Uses the Result pattern for explicit success/failure signaling.

        Args:
            conversation_id: The ID of the conversation to retrieve messages from.
            limit: Maximum number of messages to return.

        Returns:
            Result[List[MessageDict]]: On success, contains a list of message
                dictionaries. On failure, contains an Error object.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # First verify conversation exists
                cursor.execute(SQL_GET_CONVERSATION, (conversation_id,))
                if cursor.fetchone() is None:
                    error = Error.create(
                        message=f"Conversation with ID {conversation_id} not found.",
                        code="CONVERSATION_NOT_FOUND",
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.WARNING,
                        context={"conversation_id": str(conversation_id)},
                    )
                    return Result[List[MessageDict]].failure(
                        error.code,
                        error.message,
                        error.context,
                        error.category,
                        error.severity,
                    )

                cursor.execute(
                    """
                    SELECT
                        id,
                        speaker as role,
                        text as content,
                        timestamp as created_at
                    FROM conversation_messages
                    WHERE conversation_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?;
                    """,
                    (conversation_id, limit),
                )
                rows = cursor.fetchall()
                messages: List[MessageDict] = []
                for row in rows:
                    msg: MessageDict = {
                        "id": row["id"],
                        "role": row["role"],
                        "content": row["content"],
                        "created_at": row["created_at"],
                    }
                    messages.append(msg)
                return Result[List[MessageDict]].success(messages)
        except sqlite3.Error as e:
            error = Error.create(
                message=f"SQLite error getting messages: {e}",
                code="DB_SQLITE_ERROR",
                category=ErrorCategory.EXTERNAL,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "get_messages",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[List[MessageDict]].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except ConversationError as e:
            error = Error.create(
                message=f"Database connection error getting messages: {e}",
                code="DB_CONNECTION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "get_messages",
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[List[MessageDict]].failure(
                error.code, error.message, error.context, error.category, error.severity
            )
        except Exception as e:
            error = Error.create(
                message=f"Unexpected error getting messages: {e}",
                code="UNEXPECTED_GET_MSG_ERROR",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
                context={
                    "sql_operation": "get_messages",
                    "exception_type": type(e).__name__,
                    "conversation_id": str(conversation_id),
                },
            )
            return Result[List[MessageDict]].failure(
                error.code, error.message, error.context, error.category, error.severity
            )


__all__ = [
    "ConversationManager",
    "ConversationError",
    "ConversationNotFoundError",
    "ConversationDict",
    "MessageDict",
    "ModelContext",
    "ReflexiveModel",
    "LightweightModel",
    "AffectiveLexicalModel",
    "IdentityModel",
]
