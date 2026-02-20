"""Tests for word_forge.graph.graph_config module.

This module tests the graph configuration classes including GraphConfig,
various typed dictionaries, and layout/dimension types.
"""

from pathlib import Path

from word_forge.configs.config_essentials import (
    GraphColorScheme,
    GraphLayoutAlgorithm,
)
from word_forge.graph.graph_config import (
    GraphConfig,
    GraphInfoDict,
    RelationshipTupleDict,
    WordTupleDict,
)


class TestGraphConfig:
    """Tests for the GraphConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GraphConfig()
        assert config.default_layout == GraphLayoutAlgorithm.FORCE_DIRECTED
        assert config.default_color_scheme == GraphColorScheme.SEMANTIC
        assert config.min_node_size == 5
        assert config.max_node_size == 30
        assert config.min_edge_width == 0.5
        assert config.max_edge_width == 5.0
        assert config.enable_labels is True
        assert config.enable_edge_labels is False
        assert config.enable_tooltips is True
        assert config.high_quality_rendering is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GraphConfig(
            min_node_size=10,
            max_node_size=50,
            enable_labels=False,
            vis_width=1600,
            vis_height=1000,
        )
        assert config.min_node_size == 10
        assert config.max_node_size == 50
        assert config.enable_labels is False
        assert config.vis_width == 1600
        assert config.vis_height == 1000

    def test_visualization_dimensions(self):
        """Test default visualization dimensions."""
        config = GraphConfig()
        assert config.vis_width == 1200
        assert config.vis_height == 800

    def test_animation_settings(self):
        """Test default animation settings."""
        config = GraphConfig()
        assert config.animation_duration_ms == 800

    def test_limit_settings(self):
        """Test default node and edge limits."""
        config = GraphConfig()
        assert config.limit_node_count == 1000
        assert config.limit_edge_count == 2000


class TestGraphConfigActiveDimensions:
    """Tests for active dimensions configuration."""

    def test_default_active_dimensions(self):
        """Test default active dimensions include lexical, emotional, affective, and contextual."""
        config = GraphConfig()
        assert "lexical" in config.active_dimensions
        assert "emotional" in config.active_dimensions
        assert "affective" in config.active_dimensions
        assert "contextual" in config.active_dimensions
        assert len(config.active_dimensions) == 4

    def test_custom_active_dimensions(self):
        """Test custom active dimensions."""
        config = GraphConfig(active_dimensions={"lexical", "emotional"})
        assert "lexical" in config.active_dimensions
        assert "emotional" in config.active_dimensions


class TestGraphConfigDimensionWeights:
    """Tests for dimension weights configuration."""

    def test_default_dimension_weights(self):
        """Test default dimension weights."""
        config = GraphConfig()
        assert config.dimension_weights["lexical"] == 1.0
        assert config.dimension_weights["emotional"] == 0.8
        assert config.dimension_weights["affective"] == 0.6
        assert config.dimension_weights["connotative"] == 0.7
        assert config.dimension_weights["contextual"] == 0.5

    def test_all_weights_in_valid_range(self):
        """Test that all weights are between 0 and 1."""
        config = GraphConfig()
        for weight in config.dimension_weights.values():
            assert 0.0 <= weight <= 1.0


class TestGraphConfigRelationshipColors:
    """Tests for relationship color configuration."""

    def test_default_relationship_colors(self):
        """Test default relationship colors."""
        config = GraphConfig()
        assert "synonym" in config.relationship_colors
        assert "antonym" in config.relationship_colors
        assert "default" in config.relationship_colors

    def test_colors_are_hex_format(self):
        """Test that colors are in hex format."""
        config = GraphConfig()
        for color in config.relationship_colors.values():
            assert color.startswith("#")
            assert len(color) == 7

    def test_emotional_relationship_colors(self):
        """Test emotional relationship colors exist."""
        config = GraphConfig()
        assert "joy_associated" in config.emotional_relationship_colors
        assert "sadness_associated" in config.emotional_relationship_colors
        assert "anger_associated" in config.emotional_relationship_colors


class TestGraphConfigSQLTemplates:
    """Tests for SQL templates configuration."""

    def test_sql_templates_defined(self):
        """Test that SQL templates are defined."""
        config = GraphConfig()
        assert "fetch_all_words" in config.sql_templates
        assert "fetch_all_relationships" in config.sql_templates
        assert "get_emotional_relationships" in config.sql_templates

    def test_sql_templates_contain_sql(self):
        """Test that SQL templates contain valid SQL."""
        config = GraphConfig()
        assert "SELECT" in config.sql_templates["fetch_all_words"]
        assert "FROM words" in config.sql_templates["fetch_all_words"]


class TestGraphConfigSampleRelationships:
    """Tests for sample relationships configuration."""

    def test_sample_relationships_defined(self):
        """Test that sample relationships are defined."""
        config = GraphConfig()
        assert len(config.sample_relationships) > 0

    def test_sample_relationships_format(self):
        """Test sample relationships format."""
        config = GraphConfig()
        for rel in config.sample_relationships:
            assert len(rel) == 3  # (word1, word2, relationship_type)
            assert all(isinstance(s, str) for s in rel)


class TestGraphConfigMethods:
    """Tests for GraphConfig methods."""

    def test_get_visualization_path(self):
        """Test get_visualization_path cached property."""
        config = GraphConfig()
        path = config.get_visualization_path
        assert isinstance(path, Path)

    def test_get_export_path(self):
        """Test get_export_path cached property."""
        config = GraphConfig()
        path = config.get_export_path
        assert isinstance(path, Path)

    def test_get_relationship_color_known(self):
        """Test getting color for known relationship type."""
        config = GraphConfig()
        color = config.get_relationship_color("synonym")
        assert color == "#4287f5"

    def test_get_relationship_color_unknown(self):
        """Test getting color for unknown relationship type."""
        config = GraphConfig()
        color = config.get_relationship_color("unknown_type")
        assert color == config.relationship_colors["default"]


class TestGraphConfigFactoryMethods:
    """Tests for GraphConfig factory methods."""

    def test_with_emotional_relationships(self):
        """Test with_emotional_relationships factory method."""
        config = GraphConfig()
        enhanced = config.with_emotional_relationships()
        assert "emotional" in enhanced.active_dimensions
        assert "lexical" in enhanced.active_dimensions

    def test_optimize_for_interactivity(self):
        """Test optimize_for_interactivity factory method."""
        config = GraphConfig()
        interactive = config.optimize_for_interactivity()
        # Interactive mode should have lower limits for performance
        assert interactive.limit_node_count is not None

    def test_optimize_for_publication(self):
        """Test optimize_for_publication factory method."""
        config = GraphConfig()
        pub_config = config.optimize_for_publication()
        assert pub_config.high_quality_rendering is True


class TestTypedDicts:
    """Tests for the TypedDict definitions."""

    def test_word_tuple_dict(self):
        """Test WordTupleDict structure."""
        word: WordTupleDict = {"id": 1, "term": "test"}
        assert word["id"] == 1
        assert word["term"] == "test"

    def test_relationship_tuple_dict(self):
        """Test RelationshipTupleDict structure."""
        rel: RelationshipTupleDict = {
            "word_id": 1,
            "related_term": "related",
            "relationship_type": "synonym",
            "dimension": "lexical",
            "valence": 0.5,
            "arousal": 0.3,
        }
        assert rel["word_id"] == 1
        assert rel["related_term"] == "related"
        assert rel["relationship_type"] == "synonym"
        assert rel["dimension"] == "lexical"
        assert rel["valence"] == 0.5
        assert rel["arousal"] == 0.3

    def test_graph_info_dict(self):
        """Test GraphInfoDict structure."""
        info: GraphInfoDict = {
            "nodes": 100,
            "edges": 200,
            "dimensions": 2,
            "sample_nodes": [{"id": 1, "term": "test"}],
            "sample_relationships": [{"type": "synonym"}],
            "relationship_types": ["synonym", "antonym"],
        }
        assert info["nodes"] == 100
        assert info["edges"] == 200
        assert info["dimensions"] == 2


class TestGraphConfigEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_env_vars_defined(self):
        """Test that ENV_VARS class variable is defined."""
        assert hasattr(GraphConfig, "ENV_VARS")
        assert isinstance(GraphConfig.ENV_VARS, dict)


class TestGraphConfigEdgeCases:
    """Tests for edge cases in GraphConfig."""

    def test_empty_active_dimensions(self):
        """Test behavior with empty active dimensions."""
        config = GraphConfig(active_dimensions=set())
        assert len(config.active_dimensions) == 0

    def test_none_limits(self):
        """Test behavior with None limits."""
        config = GraphConfig(limit_node_count=None, limit_edge_count=None)
        assert config.limit_node_count is None
        assert config.limit_edge_count is None

    def test_custom_dimension_weights(self):
        """Test custom dimension weights."""
        custom_weights = {
            "lexical": 1.0,
            "emotional": 1.0,
            "affective": 1.0,
            "connotative": 1.0,
            "contextual": 1.0,
        }
        config = GraphConfig(dimension_weights=custom_weights)
        assert config.dimension_weights["emotional"] == 1.0
