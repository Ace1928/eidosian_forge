"""
Conversation configuration system for Word Forge.

This module defines the configuration schema for conversation management,
including status tracking, metadata handling, and retention policies for
conversation history within the Word Forge system. Adheres to Eidosian
principles for clarity, precision, and structural integrity.

Architecture:
    ┌─────────────────────┐        ┌───────────────────────────┐
    │ ConversationConfig  │───────►│ Defines Conversation Rules│
    └───────────┬─────────┘        └───────────────────────────┘
                │ Uses
    ┌───────────┴───────────────────────────────────────────┐
    │ Config Essentials (Types & Enums)                     │
    └───────────────────────────────────────────────────────┘
    ┌──────────────────────────┬────────────────────────────┐
    │ ConversationStatusMap    │ ConversationMetadataSchema │
    ├──────────────────────────┼────────────────────────────┤
    │ ConversationStatusValue  │ ConversationExportFormat   │
    ├──────────────────────────┼────────────────────────────┤
    │ ConversationRetentionPolicy│ EnvMapping                 │
    └──────────────────────────┴────────────────────────────┘
"""

from dataclasses import dataclass, field, replace
from datetime import timedelta
from functools import cached_property
from typing import Any, ClassVar, Dict, List, Set  # Added Dict, Any

# Import core types and enums for configuration structure
from word_forge.configs.config_essentials import (
    ConversationExportFormat,
    ConversationMetadataSchema,  # Represents Dict[str, Any] for metadata
    ConversationRetentionPolicy,
    ConversationStatusMap,  # Represents Dict[str, str] for status mapping
    ConversationStatusValue,  # Represents Literal[...] for specific statuses
    EnvMapping,  # Type alias for environment variable mapping dictionary
    EnvVarType,  # Union of allowed types for environment variables
)


@dataclass(frozen=True)  # Immutable configuration for safety
class ConversationConfig:
    """
    Immutable configuration for conversation management within Word Forge.

    Encapsulates rules and defaults for status tracking, metadata schema,
    history management, retention policies, and export options. Designed for
    clarity, type safety, and extensibility following Eidosian principles.

    Attributes:
        status_values: Mapping of internal status keys (lowercase strings) to
            display strings (e.g., "ACTIVE"). Ensures consistency in status
            representation.
        default_status: The default internal status key assigned to newly
            created conversations (e.g., "active"). Must be a key in
            `status_values`.
        required_metadata: A set of metadata keys (strings) that are mandatory
            for every conversation record. Used for validation.
        optional_metadata: A set of metadata keys (strings) that can be
            optionally included in conversation metadata.
        max_history_length: The maximum number of messages to retain in the
            active history of a conversation context passed to models.
        default_title_length: The default character limit used when automatically
            generating titles for conversations (if applicable).
        default_retention_policy: The default strategy applied for managing the
            retention or deletion of conversation data over time.
        export_formats: A list of supported formats (enum members) for
            exporting conversation data.
        default_export_format: The default format (enum member) used when an
            export format is not explicitly specified.
        enable_auto_archiving: A boolean flag indicating whether automatic
            archiving of inactive conversations is enabled.
        auto_archive_days: The number of days a conversation must be inactive
            (based on `updated_at`) before it becomes eligible for automatic
            archiving (if `enable_auto_archiving` is True).
        ENV_VARS: Class variable defining the mapping between environment
            variable names and the corresponding configuration attributes they
            can override. Includes the attribute name and expected type
            (as a string or Enum name) for parsing.

    Examples:
        >>> from word_forge.configs.config import default_config
        >>> config = default_config.conversation
        >>> print(config.default_status)
        active
        >>> print(config.metadata_schema)
        {'created_at', 'user_id', 'title', 'tags', 'source', 'language', 'summary', 'conversation_id'}
        >>> print(config.auto_archive_threshold)
        30 days, 0:00:00
        >>> new_config = config.with_retention_policy(ConversationRetentionPolicy.DELETE_AFTER_90_DAYS)
        >>> print(new_config.default_retention_policy)
        ConversationRetentionPolicy.DELETE_AFTER_90_DAYS

    Note:
        This class is immutable (`frozen=True`). Modifications return a new
        instance using methods like `with_retention_policy`.
    """

    # --- Status Tracking ---
    status_values: ConversationStatusMap = field(
        default_factory=lambda: {
            "active": "ACTIVE",
            "pending": "PENDING REVIEW",
            "completed": "COMPLETED",
            "archived": "ARCHIVED",
            "deleted": "DELETED",
        }
    )
    default_status: ConversationStatusValue = "active"

    # --- Metadata Schema ---
    required_metadata: Set[str] = field(
        default_factory=lambda: {"conversation_id", "created_at", "user_id"}
    )
    optional_metadata: Set[str] = field(
        default_factory=lambda: {"title", "tags", "source", "language", "summary"}
    )

    # --- History Management ---
    max_history_length: int = 100
    default_title_length: int = 50

    # --- Retention Settings ---
    default_retention_policy: ConversationRetentionPolicy = (
        ConversationRetentionPolicy.KEEP_FOREVER
    )

    # --- Export Settings ---
    export_formats: List[ConversationExportFormat] = field(
        default_factory=lambda: [
            ConversationExportFormat.JSON,
            ConversationExportFormat.MARKDOWN,
            ConversationExportFormat.TEXT,
        ]
    )
    default_export_format: ConversationExportFormat = ConversationExportFormat.JSON

    # --- Archiving ---
    enable_auto_archiving: bool = False
    auto_archive_days: int = 30

    # --- Environment Variable Overrides ---
    # Maps environment variable names to a tuple:
    # (attribute_name: str, type_representation: str | Type[Enum])
    # Type representation is used by config loading logic to parse the env var.
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_CONVERSATION_MAX_HISTORY": (
            "max_history_length",
            "int",
        ),
        "WORD_FORGE_CONVERSATION_DEFAULT_STATUS": (
            "default_status",
            "str",
        ),
        "WORD_FORGE_CONVERSATION_AUTO_ARCHIVE": (
            "enable_auto_archiving",
            "bool",
        ),
        "WORD_FORGE_CONVERSATION_RETENTION_POLICY": (
            "default_retention_policy",
            ConversationRetentionPolicy.__name__,
        ),
        "WORD_FORGE_CONVERSATION_AUTO_ARCHIVE_DAYS": (
            "auto_archive_days",
            "int",
        ),
    }

    @cached_property
    def metadata_schema(self) -> Set[str]:
        """
        Provides the complete set of allowed metadata field names.

        Combines `required_metadata` and `optional_metadata` into a single set
        for efficient validation of metadata dictionaries. This property is
        cached after the first access.

        Returns:
            Set[str]: A set containing all valid metadata field names defined
                      in this configuration.
        """
        return self.required_metadata.union(self.optional_metadata)

    @cached_property
    def auto_archive_threshold(self) -> timedelta:
        """
        Calculates the time duration threshold for automatic archiving.

        Converts the `auto_archive_days` integer into a `timedelta` object
        representing the minimum inactivity period before a conversation
        is eligible for auto-archiving. Cached for performance.

        Returns:
            timedelta: The time duration threshold for auto-archiving.
        """
        return timedelta(days=self.auto_archive_days)

    def validate_metadata(self, metadata: ConversationMetadataSchema) -> bool:
        """
        Checks if the provided metadata contains all required fields.

        Args:
            metadata: The metadata dictionary (Dict[str, Any]) to validate
                      against the `required_metadata` set.

        Returns:
            bool: True if all keys specified in `required_metadata` are present
                  in the `metadata` dictionary, False otherwise.
        """
        return self.required_metadata.issubset(metadata.keys())

    def get_missing_fields(self, metadata: ConversationMetadataSchema) -> Set[str]:
        """
        Identifies which required metadata fields are missing from a given dictionary.

        Args:
            metadata: The metadata dictionary (Dict[str, Any]) to check against
                      the `required_metadata` set.

        Returns:
            Set[str]: A set containing the names of required fields that are
                      not present as keys in the `metadata` dictionary. Returns
                      an empty set if all required fields are present or if
                      metadata is not a dictionary.
        """
        return self.required_metadata.difference(metadata.keys())

    def create_default_metadata(self) -> Dict[str, Any]:  # Changed return type hint
        """
        Generates a template metadata dictionary with required fields set to None.

        Provides a basic structure containing all mandatory metadata keys,
        useful for initializing new conversation metadata or understanding the
        required schema.

        Returns:
            Dict[str, Any]: A dictionary where keys are the required
                            metadata fields and all values are `None`.
        """
        default_meta: Dict[str, Any] = {field: None for field in self.required_metadata}
        return default_meta

    def with_retention_policy(
        self, policy: ConversationRetentionPolicy
    ) -> "ConversationConfig":
        """
        Creates a new `ConversationConfig` instance with an updated retention policy.

        Leverages the immutability of the dataclass. Returns a new object
        with only the `default_retention_policy` changed, leaving the original
        instance unmodified. This adheres to functional programming principles.

        Args:
            policy: The new `ConversationRetentionPolicy` enum member to set.

        Returns:
            ConversationConfig: A new, independent `ConversationConfig` instance
                                reflecting the updated retention policy.
        """
        return replace(self, default_retention_policy=policy)

    def is_valid_status(self, status_key: str) -> bool:
        """
        Checks if a given string is a valid internal status key.

        Validates against the keys defined in the `status_values` dictionary.
        This checks the internal representation (e.g., "active"), not the
        display string (e.g., "ACTIVE").

        Args:
            status_key: The internal status string (lowercase) to validate.

        Returns:
            bool: True if `status_key` is a defined key in the
                  `status_values` dictionary, False otherwise.
        """
        return status_key in self.status_values

    def is_active_status(self, status_key: ConversationStatusValue) -> bool:
        """
        Determines if a given internal status key represents an active/ongoing state.

        Considers statuses like "active" and "pending" as representing
        non-terminal conversation states where further interaction or processing
        might occur.

        Args:
            status_key: The `ConversationStatusValue` (internal key) to check.

        Returns:
            bool: True if the status key is considered active ("active" or "pending"),
                  False otherwise.
        """
        return status_key in ("active", "pending")


# ==========================================
# Module Exports
# ==========================================

# Explicitly define the public interface of this module
__all__ = [
    "ConversationConfig",
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    "ConversationRetentionPolicy",
    "ConversationExportFormat",
    "EnvMapping",
    "EnvVarType",
]
