"""
Parser configuration system for Word Forge lexical data processing.

This module defines the configuration schema for lexical parsers, including
resource paths, model settings, and data source management. It provides
a consistent interface for accessing resources across the Word Forge system.

Architecture:
    ┌─────────────────────┐
    │    ParserConfig     │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Paths│Model│Data │Sources│Error │
    └─────┴─────┴─────┴───────┴─────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Dict, Optional

from word_forge.configs.config_essentials import DATA_ROOT, ConfigError
from word_forge.configs.config_types import EnvMapping


# Custom exception for resource handling
class ResourceNotFoundError(ConfigError):
    """Exception raised when a requested resource cannot be found."""

    pass


@dataclass
class ParserConfig:
    """
    Configuration for lexical data parser.

    Controls data sources, model settings, and resource paths for
    the system's parsing components. Provides methods for accessing
    and validating lexical resources with strict typing and error handling.

    Attributes:
        data_dir: Base directory for lexical resources
        enable_model: Whether to use language models for examples
        model_name: Custom language model name (None uses vectorizer's model)
        resource_paths: Paths to various lexical resources, relative to data_dir
        ENV_VARS: Mapping of environment variables to config attributes

    Examples:
        >>> from word_forge.config import config
        >>> # Get resource path
        >>> thesaurus_path = config.parser.get_full_resource_path("thesaurus")
        >>> # Check if resource exists
        >>> if config.parser.resource_exists("openthesaurus"):
        ...     # Process resource
        ...     pass
        >>> # Create configuration with custom model
        >>> custom_config = config.parser.with_custom_model("gpt2-medium")
    """

    # Base directory for lexical resources
    data_dir: str = str(DATA_ROOT)

    # Control language model usage for examples
    enable_model: bool = True

    # Custom language model name (None = use vectorizer's model)
    model_name: Optional[str] = None

    # Resource paths relative to data_dir
    resource_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "openthesaurus": "openthesaurus.jsonl",
            "odict": "odict.json",
            "dbnary": "dbnary.ttl",
            "opendict": "opendict.json",
            "thesaurus": "thesaurus.jsonl",
        }
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_DATA_DIR": ("data_dir", str),
        "WORD_FORGE_ENABLE_MODEL": ("enable_model", bool),
        "WORD_FORGE_PARSER_MODEL": ("model_name", str),
    }

    @cached_property
    def data_dir_path(self) -> Path:
        """
        Get data directory as a Path object.

        Returns:
            Path: Absolute path to the data directory
        """
        return Path(self.data_dir)

    def get_full_resource_path(self, resource_name: str) -> Path:
        """
        Get absolute path for a resource.

        Args:
            resource_name: The name of the resource as defined in resource_paths

        Returns:
            Path: Absolute path to the resource

        Raises:
            ResourceNotFoundError: If the resource name is not configured

        Examples:
            >>> config = ParserConfig()
            >>> path = config.get_full_resource_path("thesaurus")
            >>> path.name
            'thesaurus.jsonl'
        """
        if resource_name not in self.resource_paths:
            raise ResourceNotFoundError(
                f"Unknown resource: '{resource_name}'. "
                f"Available resources: {', '.join(sorted(self.resource_paths.keys()))}"
            )
        return self.data_dir_path / self.resource_paths[resource_name]

    def resource_exists(self, resource_name: str) -> bool:
        """
        Check if a resource exists on the filesystem.

        Args:
            resource_name: The name of the resource to check

        Returns:
            bool: True if the resource exists, False otherwise

        Raises:
            ResourceNotFoundError: If the resource name is not configured
        """
        return self.get_full_resource_path(resource_name).exists()

    def get_all_resource_paths(self) -> Dict[str, Path]:
        """
        Get absolute paths for all configured resources.

        Returns:
            Dict[str, Path]: Mapping of resource names to absolute paths
        """
        return {
            name: self.data_dir_path / path
            for name, path in self.resource_paths.items()
        }

    def with_custom_model(self, model_name: str) -> ParserConfig:
        """
        Create a new configuration with a custom language model.

        Args:
            model_name: Name of the language model to use

        Returns:
            ParserConfig: New configuration with the specified model

        Examples:
            >>> config = ParserConfig()
            >>> custom = config.with_custom_model("gpt2-medium")
            >>> custom.model_name
            'gpt2-medium'
            >>> custom.enable_model
            True
        """
        return ParserConfig(
            data_dir=self.data_dir,
            enable_model=True,  # Force enable when specifying a model
            model_name=model_name,
            resource_paths=self.resource_paths.copy(),
        )


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    "ParserConfig",
    "ResourceNotFoundError",
]
