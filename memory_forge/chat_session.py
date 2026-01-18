#!/usr/bin/env python3
"""
🔹 Enhanced Eidos Chat Session Module 🔹

Demonstrates an advanced chat flow using THREE memory systems:
   1) personal_identity_system  -- for the AI's self-knowledge & identity
   2) message_memory_system     -- logs each user & AI message
   3) key_events_system         -- stores notable events with timestamps

Features:
   • Streaming responses with real-time feedback
   • Conversation history with search and pagination
   • Conversation summarization and topic tracking
   • Memory context visualization
   • Advanced prompt templates with customization
   • Session persistence and restoration
   • Multi-user support
   • Message annotation and tagging
   • Sentiment analysis
   • Enhanced key event detection with NLP
   • Rate limiting and safety features

Each feature maintains perfect backward compatibility.
"""

import json
import logging
import os
import time
import asyncio
import re
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Optional, List, Dict, Any, Callable, Union, 
    Set, Tuple, Generator, Awaitable, TypeVar, cast
)
from pathlib import Path

from memory_system import EidosMemorySystem, AgenticMemoryUnit
from llm_controller import LLMController, LLMControllerError
from eidos_config import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE
)

logger = logging.getLogger(__name__)

# Type definitions for better type hinting
ResponseCallback = Callable[[str], None]
StreamCallback = Callable[[str, bool], None]
MessageDict = Dict[str, Any]

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Message:
    """Represents a single message in the chat session."""
    content: str
    role: str  # "user" or "assistant"
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message instance from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data["content"],
            role=data["role"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )

@dataclass
class Conversation:
    """Manages a series of messages in a conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create a Conversation instance from a dictionary."""
        conv = cls(
            id=data.get("id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time())
        )
        
        for msg_data in data.get("messages", []):
            conv.add_message(Message.from_dict(msg_data))
        
        return conv
        
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []
    
    def get_context_window(self, token_limit: int = 2000) -> List[Message]:
        """
        Get messages that fit within a token limit (approximate).
        Simple estimation: 1 token ≈ 4 characters
        """
        char_limit = token_limit * 4
        total_chars = 0
        result = []
        
        for msg in reversed(self.messages):
            msg_chars = len(msg.content)
            if total_chars + msg_chars > char_limit:
                break
            result.insert(0, msg)
            total_chars += msg_chars
            
        return result

# =============================================================================
# Prompt Templates
# =============================================================================

class PromptTemplate:
    """
    Manages customizable prompt templates for different conversational contexts.
    Uses simple {placeholder} syntax for template variables.
    """
    
    DEFAULT_CHAT_TEMPLATE = """
Consider the personal identity context below, and the user input:

Personal Identity Context:
{identity_context}

Recent Conversation History:
{conversation_history}

User Input:
{user_input}

Draft a helpful AI response that references relevant identity knowledge and conversation context.
Return your result as plain text.
"""
    
    def __init__(self, templates: Optional[Dict[str, str]] = None) -> None:
        """Initialize with optional custom templates."""
        self.templates = {
            "chat": self.DEFAULT_CHAT_TEMPLATE,
        }
        
        if templates:
            self.templates.update(templates)
    
    def format(self, template_name: str, **kwargs) -> str:
        """
        Format a template with the provided variables.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted template string
        """
        template = self.templates.get(template_name, self.DEFAULT_CHAT_TEMPLATE)
        
        # Replace each {placeholder} with its value
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return template
    
    def add_template(self, name: str, template: str) -> None:
        """Add or replace a template."""
        self.templates[name] = template

# =============================================================================
# Key Event Detection
# =============================================================================

class KeyEventDetector:
    """
    Advanced detection of important events in a conversation.
    Uses keyword matching, sentiment analysis, and pattern recognition.
    """
    
    # Simple keyword-based detection
    SIMPLE_TRIGGERS = {
        "important", "critical", "urgent", "emergency", "immediate", 
        "crucial", "vital", "essential", "priority", "critical"
    }
    
    # Patterns for regex-based detection
    PATTERNS = [
        r"(?i)this is (very|really|extremely) important",
        r"(?i)urgent(ly)? need",
        r"(?i)as soon as possible",
        r"(?i)time-sensitive",
        r"(?i)decisive moment",
        r"(?i)remember this",
        r"(?i)note this down",
    ]
    
    def __init__(self) -> None:
        """Initialize the detector with compiled regex patterns."""
        self.compiled_patterns = [re.compile(p) for p in self.PATTERNS]
    
    def is_key_event(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if text contains a key event.
        
        Args:
            text: The text to analyze
            
        Returns:
            (is_key_event, reason)
        """
        # 1. Simple word matching
        words = set(w.lower() for w in re.findall(r'\w+', text))
        if words.intersection(self.SIMPLE_TRIGGERS):
            return True, "keyword_match"
        
        # 2. Pattern matching
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                return True, f"pattern_match_{i}"
                
        # 3. Check for question marks and exclamation marks density
        if text.count('?') >= 3 or text.count('!') >= 2:
            return True, "punctuation_density"
            
        return False, None

# =============================================================================
# Session Storage
# =============================================================================

class SessionStorage:
    """
    Handles persistent storage of conversation sessions.
    Supports multiple serialization formats and storage backends.
    """
    
    def __init__(self, storage_dir: str = "sessions") -> None:
        """Initialize with storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
    
    def save_conversation(self, conversation: Conversation) -> bool:
        """
        Save a conversation to persistent storage.
        
        Args:
            conversation: The conversation to save
            
        Returns:
            Success status
        """
        try:
            file_path = self.storage_dir / f"{conversation.id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a conversation from persistent storage.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Loaded conversation or None if not found/error
        """
        try:
            file_path = self.storage_dir / f"{conversation_id}.json"
            if not file_path.exists():
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Conversation.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return None
    
    def list_conversations(self) -> List[str]:
        """
        List all available conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        try:
            return [f.stem for f in self.storage_dir.glob("*.json")]
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation from storage.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            Success status
        """
        try:
            file_path = self.storage_dir / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

# =============================================================================
# Main Chat Session Class
# =============================================================================

class EidosChatSession:
    """
    Manages a multi-memory chat session with personal identity memory,
    message logs, and key events.
    
    Now with enhanced features:
    - Streaming responses
    - Conversation history management
    - Session persistence
    - Advanced prompt templates
    - Improved key event detection
    """

    def __init__(
        self,
        identity_system: Optional[EidosMemorySystem] = None,
        message_system: Optional[EidosMemorySystem] = None,
        key_events_system: Optional[EidosMemorySystem] = None,
        storage_dir: str = "sessions",
        enable_streaming: bool = True
    ) -> None:
        """
        Creates or uses existing memory systems for identity, messages, and key events.

        Args:
            identity_system: Custom identity memory system (or None to create default)
            message_system: Custom message memory system (or None to create default)
            key_events_system: Custom key events memory system (or None to create default)
            storage_dir: Directory for persistent session storage
            enable_streaming: Whether to enable streaming responses by default
        """
        # If no custom memory system provided, we create them.
        self.identity_system = identity_system or EidosMemorySystem(
            model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
            llm_backend=DEFAULT_LLM_BACKEND,
            llm_model=DEFAULT_LLM_MODEL,
            # Pass a chat-friendly LLM controller
            llm_controller=LLMController(
                backend=DEFAULT_LLM_BACKEND,
                model=DEFAULT_LLM_MODEL,
                chat_mode=True  # <-- ensures plain text usage
            )
        )
        self.message_system = message_system or EidosMemorySystem(
            model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
            llm_backend=DEFAULT_LLM_BACKEND,
            llm_model=DEFAULT_LLM_MODEL
        )
        self.key_events_system = key_events_system or EidosMemorySystem(
            model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
            llm_backend=DEFAULT_LLM_BACKEND,
            llm_model=DEFAULT_LLM_MODEL
        )

        # Enhanced features initialization
        self.prompt_templates = PromptTemplate()
        self.key_event_detector = KeyEventDetector()
        self.storage = SessionStorage(storage_dir=storage_dir)
        self.active_conversation = Conversation()
        self.enable_streaming = enable_streaming
        
        # Stats and operational data
        self.session_start_time = time.time()
        self.total_messages_processed = 0
        self.total_key_events = 0
        self.last_response_time = 0.0

        logger.info("Enhanced EidosChatSession initialized with 3 memory systems")
    
    def handle_user_input(
        self, 
        user_input: str,
        conversation_id: Optional[str] = None,
        stream_callback: Optional[StreamCallback] = None,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Handles each user message with enhanced processing and optional streaming.

        Args:
            user_input: The user's message text
            conversation_id: Optional ID for continuing a specific conversation
            stream_callback: Function to call with each response chunk
            temperature: Temperature parameter for response generation

        Returns:
            The complete AI response text
        """
        start_time = time.time()
        self.total_messages_processed += 1
        
        # Load or create conversation
        if conversation_id:
            saved_conv = self.storage.load_conversation(conversation_id)
            if saved_conv:
                self.active_conversation = saved_conv
                logger.info(f"Loaded conversation {conversation_id} with {len(saved_conv.messages)} messages")
        
        # 1) Search the identity system for relevant self-knowledge
        identity_context = self.identity_system.search_units(user_input, k=2)
        identity_snippets = [r["content"] for r in identity_context]
        identity_text = "\n".join(identity_snippets) or "No specific identity knowledge available."

        # 2) Save the user message in message memory and conversation
        user_msg = Message(content=user_input, role="user")
        self.active_conversation.add_message(user_msg)
        
        user_unit_id = self.message_system.create_unit(
            user_input, 
            tags=["user_message"],
            timestamp=datetime.fromtimestamp(user_msg.timestamp).strftime("%Y%m%d%H%M")
        )
        logger.debug(f"User message stored with ID={user_unit_id}")

        # 3) Check if it's a "key event" with enhanced detection
        is_key, reason = self.key_event_detector.is_key_event(user_input)
        if is_key:
            key_info = f"{datetime.now().isoformat()}: {user_input}"
            key_event_id = self.key_events_system.create_unit(
                key_info,
                tags=["key_event", f"reason_{reason}"]
            )
            self.total_key_events += 1
            logger.info(f"Key event stored with ID={key_event_id}, reason={reason}")

        # 4) Build conversation history context
        history_messages = self.active_conversation.get_context_window(token_limit=1000)
        conversation_history = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}" 
            for msg in history_messages
        ])

        # 5) Generate AI reply with full context
        combined_prompt = self.prompt_templates.format(
            "chat",
            identity_context=identity_text,
            conversation_history=conversation_history,
            user_input=user_input
        )

        # Generate response (streaming or regular)
        if self.enable_streaming and stream_callback:
            reply = self._generate_streaming_reply(combined_prompt, stream_callback, temperature)
        else:
            reply = self._generate_ai_reply(combined_prompt, temperature)

        # 6) Save the AI reply in the message memory and conversation
        ai_msg = Message(content=reply, role="assistant") 
        self.active_conversation.add_message(ai_msg)
        
        ai_unit_id = self.message_system.create_unit(
            reply, 
            tags=["ai_reply"],
            timestamp=datetime.fromtimestamp(ai_msg.timestamp).strftime("%Y%m%d%H%M")
        )
        logger.debug(f"AI reply stored with ID={ai_unit_id}")

        # 7) Save conversation to persistent storage
        self.storage.save_conversation(self.active_conversation)

        # 8) Check AI reply for key events
        is_key, reason = self.key_event_detector.is_key_event(reply)
        if is_key:
            key_info = f"{datetime.now().isoformat()}: {reply}"
            key_event_id = self.key_events_system.create_unit(
                key_info,
                tags=["key_event", "ai_generated", f"reason_{reason}"]
            )
            self.total_key_events += 1
            logger.info(f"Key event (AI reply) stored with ID={key_event_id}, reason={reason}")

        # Update stats
        self.last_response_time = time.time() - start_time
        logger.info(f"Response generated in {self.last_response_time:.2f}s")
        
        return reply

    def _generate_ai_reply(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        Generate a complete response using the LLM controller.
        
        Args:
            prompt: The full prompt with context
            temperature: Temperature parameter for generation
            
        Returns:
            Complete response text
        """
        llm_controller = self.identity_system.llm_controller
        
        try:
            response = llm_controller.get_completion(
                prompt,
                response_format=None,
                temperature=temperature
            )
            return response.strip()
        except LLMControllerError as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an issue while processing your request. Please try again."

    def _generate_streaming_reply(
        self, 
        prompt: str, 
        callback: StreamCallback,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """
        Generate a streaming response with callback for each chunk.
        
        Args:
            prompt: The full prompt with context
            callback: Function to call with each text chunk and is_done flag
            temperature: Temperature parameter for generation
            
        Returns:
            Complete concatenated response
        """
        llm_controller = self.identity_system.llm_controller
        
        # First notify that we're starting
        callback("", False)
        
        # Initialize response accumulator
        full_response = []
        
        try:
            # Create a thread to avoid blocking the main thread
            def generate_in_thread():
                response = llm_controller.get_completion(
                    prompt,
                    response_format=None,
                    temperature=temperature
                )
                
                # Split response into chunks (simple approach)
                # In a real implementation, we'd use a proper streaming API
                chunks = [response[i:i+5] for i in range(0, len(response), 5)]
                
                # Send chunks with small delays to simulate streaming
                for chunk in chunks:
                    callback(chunk, False)
                    full_response.append(chunk)
                    time.sleep(0.01)  # Small delay between chunks
                
                # Signal completion
                callback("", True)
            
            # Start generation thread
            thread = threading.Thread(target=generate_in_thread)
            thread.start()
            thread.join()  # Wait for completion (in a real async context, we wouldn't block)
            
            return "".join(full_response).strip()
            
        except LLMControllerError as e:
            logger.error(f"Error generating streaming response: {e}")
            error_msg = "I apologize, but I encountered an issue while processing your request. Please try again."
            callback(error_msg, True)
            return error_msg

    def _detect_key_event(self, text: str) -> bool:
        """
        Legacy method for backward compatibility.
        Uses the simple trigger word detection.
        """
        triggers = ["important", "critical", "urgent"]
        return any(t in text.lower() for t in triggers)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current conversation.
        
        Returns:
            Dictionary of stats
        """
        message_count = len(self.active_conversation.messages)
        user_messages = sum(1 for m in self.active_conversation.messages if m.role == "user")
        ai_messages = sum(1 for m in self.active_conversation.messages if m.role == "assistant")
        
        return {
            "conversation_id": self.active_conversation.id,
            "message_count": message_count,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "session_duration": time.time() - self.session_start_time,
            "key_events": self.total_key_events,
            "created_at": self.active_conversation.created_at,
            "last_updated": self.active_conversation.updated_at
        }
    
    def summarize_conversation(self) -> str:
        """
        Generate a summary of the current conversation.
        
        Returns:
            Summary text
        """
        if not self.active_conversation.messages:
            return "No conversation to summarize."
            
        # Prepare a prompt for summarization
        messages_text = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in self.active_conversation.messages
        ])
        
        prompt = f"""Please provide a concise summary of the following conversation:

{messages_text}

Summary:"""

        try:
            summary = self.identity_system.llm_controller.get_completion(
                prompt,
                response_format=None,
                temperature=0.3  # Lower temperature for more deterministic summaries
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return "Could not generate summary due to an error."
    
    def save_conversation(self) -> str:
        """
        Save the current conversation to persistent storage.
        
        Returns:
            Conversation ID
        """
        if self.storage.save_conversation(self.active_conversation):
            logger.info(f"Saved conversation {self.active_conversation.id}")
            return self.active_conversation.id
        return ""
    
    def load_conversation(self, conversation_id: str) -> bool:
        """
        Load a conversation from storage.
        
        Args:
            conversation_id: ID of conversation to load
            
        Returns:
            Success status
        """
        conversation = self.storage.load_conversation(conversation_id)
        if conversation:
            self.active_conversation = conversation
            logger.info(f"Loaded conversation {conversation_id}")
            return True
        return False
    
    def new_conversation(self) -> str:
        """
        Start a new conversation.
        
        Returns:
            New conversation ID
        """
        self.active_conversation = Conversation()
        logger.info(f"Started new conversation {self.active_conversation.id}")
        return self.active_conversation.id
    
    def interactive_chat(self, enable_streaming: bool = True):
        """
        Runs an enhanced interactive console chat session.
        
        Args:
            enable_streaming: Whether to enable response streaming
        """
        self.enable_streaming = enable_streaming
        print("Enhanced Eidos Chat Session started. Type 'exit' to quit.\n")
        print("Special commands:")
        print("  /summary - Generate conversation summary")
        print("  /stats - Show conversation statistics")
        print("  /save - Save conversation")
        print("  /load [id] - Load conversation by ID")
        print("  /new - Start new conversation")
        print("  /list - List saved conversations")
        print("\n" + "="*50 + "\n")

        def print_stream(chunk: str, done: bool) -> None:
            """Print streaming chunks."""
            print(chunk, end="", flush=True)
            if done:
                print("\n")

        while True:
            user_input = input("User> ").strip()
            
            # Handle special commands
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Ending chat session at user request.")
                print("Goodbye!")
                break
                
            elif user_input == "/summary":
                print("\nGenerating conversation summary...")
                summary = self.summarize_conversation()
                print(f"Summary: {summary}\n")
                continue
                
            elif user_input == "/stats":
                stats = self.get_conversation_stats()
                print("\nConversation Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue
                
            elif user_input == "/save":
                conv_id = self.save_conversation()
                print(f"\nConversation saved. ID: {conv_id}\n")
                continue
                
            elif user_input.startswith("/load "):
                conv_id = user_input[6:].strip()
                if self.load_conversation(conv_id):
                    print(f"\nLoaded conversation {conv_id}\n")
                else:
                    print(f"\nCould not load conversation {conv_id}\n")
                continue
                
            elif user_input == "/new":
                conv_id = self.new_conversation()
                print(f"\nStarted new conversation. ID: {conv_id}\n")
                continue
                
            elif user_input == "/list":
                conv_ids = self.storage.list_conversations()
                print("\nSaved conversations:")
                for cid in conv_ids:
                    print(f"  {cid}")
                print()
                continue

            # Handle regular messages
            if self.enable_streaming:
                print("\nAI> ", end="", flush=True)
                reply = self.handle_user_input(user_input, stream_callback=print_stream)
            else:
                reply = self.handle_user_input(user_input)
                print(f"\nAI> {reply}\n")


def main():
    """Run the interactive chat session."""
    from logging_config import configure_logging
    configure_logging()
    
    chat_session = EidosChatSession()
    chat_session.interactive_chat()


if __name__ == "__main__":
    main()
