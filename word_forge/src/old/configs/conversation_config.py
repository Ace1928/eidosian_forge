"""
Conversation configuration system for Word Forge.

This module defines the configuration schema for conversation management,
including status tracking, metadata handling, and retention policies for
conversation history within the Word Forge system.

Architecture:
    ┌─────────────────────┐
    │ ConversationConfig  │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Stats│Meta │Hist │Reten  │Exp  │
    └─────┴─────┴─────┴───────┴─────┘
"""

from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property
from typing import ClassVar, Dict, List, Set

from word_forge.configs.config_essentials import (
    ConversationExportFormat,
    ConversationMetadataSchema,
    ConversationRetentionPolicy,
    ConversationStatusMap,
    ConversationStatusValue,
)
from word_forge.configs.config_types import EnvMapping


@dataclass
class ConversationConfig:
    """
    Configuration for conversation management.

    Controls status tracking, metadata schema, history management,
    retention policies, and export options for conversations.

    Attributes:
        status_values: Status codes and their human-readable descriptions
        default_status: Default status for new conversations
        required_metadata: Required metadata fields for each conversation
        optional_metadata: Optional metadata fields that can be attached
        max_history_length: Maximum number of messages in conversation history
        default_title_length: Default length for auto-generated titles
        default_retention_policy: Default policy for conversation retention
        export_formats: Available formats for exporting conversations
        default_export_format: Default format for exporting conversations
        enable_auto_archiving: Whether to automatically archive old conversations
        auto_archive_days: Days after which to auto-archive inactive conversations

    Usage:
        from word_forge.config import config

        # Access settings
        status = config.conversation.status_values["active"]

        # Check if metadata field is required
        is_required = "user_id" in config.conversation.required_metadata

        # Get default retention policy
        policy = config.conversation.default_retention_policy
    """

    # Status tracking
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

    # Metadata schema
    required_metadata: Set[str] = field(
        default_factory=lambda: {"conversation_id", "created_at", "user_id"}
    )
    optional_metadata: Set[str] = field(
        default_factory=lambda: {"title", "tags", "source", "language", "summary"}
    )

    # History management
    max_history_length: int = 100
    default_title_length: int = 50

    # Retention settings
    default_retention_policy: ConversationRetentionPolicy = (
        ConversationRetentionPolicy.KEEP_FOREVER
    )

    # Export settings
    export_formats: List[ConversationExportFormat] = field(
        default_factory=lambda: [
            ConversationExportFormat.JSON,
            ConversationExportFormat.MARKDOWN,
            ConversationExportFormat.TEXT,
        ]
    )
    default_export_format: ConversationExportFormat = ConversationExportFormat.JSON

    # Archiving
    enable_auto_archiving: bool = False
    auto_archive_days: int = 30

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_CONVERSATION_MAX_HISTORY": ("max_history_length", int),
        "WORD_FORGE_CONVERSATION_DEFAULT_STATUS": ("default_status", str),
        "WORD_FORGE_CONVERSATION_AUTO_ARCHIVE": ("enable_auto_archiving", bool),
        "WORD_FORGE_CONVERSATION_RETENTION_POLICY": (
            "default_retention_policy",
            ConversationRetentionPolicy,
        ),
    }

    @cached_property
    def metadata_schema(self) -> Set[str]:
        """
        Complete metadata schema combining required and optional fields.

        Returns:
            Set[str]: All valid metadata field names
        """
        return self.required_metadata.union(self.optional_metadata)

    @cached_property
    def auto_archive_threshold(self) -> timedelta:
        """
        Time threshold for auto-archiving as a timedelta.

        Returns:
            timedelta: Time period after which conversations are archived
        """
        return timedelta(days=self.auto_archive_days)

    def validate_metadata(self, metadata: Dict[str, any]) -> bool:
        """
        Validate if provided metadata contains all required fields.

        Args:
            metadata: Dictionary of metadata to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return all(field in metadata for field in self.required_metadata)

    def get_missing_fields(self, metadata: Dict[str, any]) -> Set[str]:
        """
        Get required fields missing from provided metadata.

        Args:
            metadata: Dictionary of metadata to check

        Returns:
            Set[str]: Set of missing required field names
        """
        return self.required_metadata - set(metadata.keys())

    def create_default_metadata(self) -> ConversationMetadataSchema:
        """
        Create default metadata structure with empty fields.

        Returns:
            ConversationMetadataSchema: Template with all required fields
        """
        return {field: None for field in self.required_metadata}

    def with_retention_policy(
        self, policy: ConversationRetentionPolicy
    ) -> "ConversationConfig":
        """
        Create new configuration with specified retention policy.

        Args:
            policy: New retention policy to apply

        Returns:
            ConversationConfig: New configuration instance
        """
        return ConversationConfig(
            status_values=self.status_values,
            default_status=self.default_status,
            required_metadata=self.required_metadata,
            optional_metadata=self.optional_metadata,
            max_history_length=self.max_history_length,
            default_title_length=self.default_title_length,
            default_retention_policy=policy,
            export_formats=self.export_formats,
            default_export_format=self.default_export_format,
            enable_auto_archiving=self.enable_auto_archiving,
            auto_archive_days=self.auto_archive_days,
        )

    def is_valid_status(self, status: str) -> bool:
        """
        Check if a status value is valid in the current configuration.

        Args:
            status: Status value to validate

        Returns:
            bool: True if status is valid, False otherwise
        """
        return status in self.status_values

    def is_active_status(self, status: str) -> bool:
        """
        Check if a status indicates an active conversation.

        Args:
            status: Status value to check

        Returns:
            bool: True if conversation is considered active
        """
        return status == "active" or status == "pending"


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    "ConversationConfig",
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    "ConversationRetentionPolicy",
    "ConversationExportFormat",
]
