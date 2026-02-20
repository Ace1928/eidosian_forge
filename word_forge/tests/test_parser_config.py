"""Test suite for parser configuration module.

This module tests the ParserConfig class and related functionality including:
- Configuration initialization and defaults
- Resource path handling
- Environment variable overrides
- Custom model configuration
"""

from pathlib import Path

import pytest
from word_forge.parser.parser_config import ParserConfig, ResourceNotFoundError


class TestParserConfigInitialization:
    """Tests for ParserConfig initialization and defaults."""

    def test_default_initialization(self):
        """Test that ParserConfig initializes with default values."""
        config = ParserConfig()
        assert config.enable_model is True
        assert config.model_name is None
        assert isinstance(config.resource_paths, dict)
        assert "openthesaurus" in config.resource_paths
        assert "thesaurus" in config.resource_paths

    def test_custom_data_dir(self, tmp_path: Path):
        """Test initialization with custom data directory."""
        config = ParserConfig(data_dir=str(tmp_path))
        assert config.data_dir == str(tmp_path)
        assert config.data_dir_path == tmp_path

    def test_disable_model(self):
        """Test initialization with model disabled."""
        config = ParserConfig(enable_model=False)
        assert config.enable_model is False

    def test_custom_model_name(self):
        """Test initialization with custom model name."""
        config = ParserConfig(model_name="gpt2-medium")
        assert config.model_name == "gpt2-medium"


class TestResourcePaths:
    """Tests for resource path handling."""

    def test_get_full_resource_path_valid(self, tmp_path: Path):
        """Test getting full path for a valid resource."""
        config = ParserConfig(data_dir=str(tmp_path))
        path = config.get_full_resource_path("thesaurus")
        assert path == tmp_path / "thesaurus.jsonl"

    def test_get_full_resource_path_invalid(self):
        """Test that invalid resource name raises ResourceNotFoundError."""
        config = ParserConfig()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            config.get_full_resource_path("nonexistent_resource")
        assert "Unknown resource" in str(exc_info.value)
        assert "nonexistent_resource" in str(exc_info.value)

    def test_get_all_resource_paths(self, tmp_path: Path):
        """Test getting all resource paths."""
        config = ParserConfig(data_dir=str(tmp_path))
        all_paths = config.get_all_resource_paths()
        assert isinstance(all_paths, dict)
        assert len(all_paths) == len(config.resource_paths)
        for name, path in all_paths.items():
            assert isinstance(path, Path)
            assert path.parent == tmp_path

    def test_resource_exists_true(self, tmp_path: Path):
        """Test resource_exists returns True for existing files."""
        # Create a temporary resource file
        thesaurus_file = tmp_path / "thesaurus.jsonl"
        thesaurus_file.touch()

        config = ParserConfig(data_dir=str(tmp_path))
        assert config.resource_exists("thesaurus") is True

    def test_resource_exists_false(self, tmp_path: Path):
        """Test resource_exists returns False for missing files."""
        config = ParserConfig(data_dir=str(tmp_path))
        assert config.resource_exists("thesaurus") is False


class TestCustomModel:
    """Tests for custom model configuration."""

    def test_with_custom_model(self):
        """Test creating config with custom model."""
        base_config = ParserConfig(enable_model=False)
        custom_config = base_config.with_custom_model("gpt2-medium")

        # Custom config should have model enabled
        assert custom_config.enable_model is True
        assert custom_config.model_name == "gpt2-medium"

        # Original config should be unchanged
        assert base_config.enable_model is False
        assert base_config.model_name is None

    def test_with_custom_model_preserves_paths(self, tmp_path: Path):
        """Test that with_custom_model preserves resource paths."""
        custom_paths = {"custom": "custom.json"}
        base_config = ParserConfig(data_dir=str(tmp_path), resource_paths=custom_paths.copy())
        custom_config = base_config.with_custom_model("test-model")

        assert custom_config.data_dir == str(tmp_path)
        assert custom_config.resource_paths == custom_paths


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_env_var_mapping_defined(self):
        """Test that ENV_VARS class variable is properly defined."""
        assert hasattr(ParserConfig, "ENV_VARS")
        assert isinstance(ParserConfig.ENV_VARS, dict)
        assert "WORD_FORGE_DATA_DIR" in ParserConfig.ENV_VARS
        assert "WORD_FORGE_ENABLE_MODEL" in ParserConfig.ENV_VARS
        assert "WORD_FORGE_PARSER_MODEL" in ParserConfig.ENV_VARS

    def test_env_var_mapping_structure(self):
        """Test that ENV_VARS mappings have correct structure."""
        for env_var, mapping in ParserConfig.ENV_VARS.items():
            assert isinstance(env_var, str)
            assert isinstance(mapping, tuple)
            assert len(mapping) == 2
            attr_name, attr_type = mapping
            assert isinstance(attr_name, str)
            # attr_type should be a type
            assert attr_type in (str, bool, int, float)


class TestDataDirPath:
    """Tests for data_dir_path cached property."""

    def test_data_dir_path_is_path(self, tmp_path: Path):
        """Test that data_dir_path returns a Path object."""
        config = ParserConfig(data_dir=str(tmp_path))
        assert isinstance(config.data_dir_path, Path)

    def test_data_dir_path_cached(self, tmp_path: Path):
        """Test that data_dir_path is cached."""
        config = ParserConfig(data_dir=str(tmp_path))
        path1 = config.data_dir_path
        path2 = config.data_dir_path
        # Same object reference due to caching
        assert path1 is path2


class TestResourceNotFoundError:
    """Tests for ResourceNotFoundError exception."""

    def test_exception_message(self):
        """Test ResourceNotFoundError message formatting."""
        error = ResourceNotFoundError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_inheritance(self):
        """Test that ResourceNotFoundError inherits from ConfigError."""
        from word_forge.configs.config_essentials import ConfigError

        error = ResourceNotFoundError("test")
        assert isinstance(error, ConfigError)
