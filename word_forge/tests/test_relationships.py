"""Tests for word_forge.relationships module.

This module tests the relationship type definitions and helper functions
used throughout the Word Forge lexical network.
"""

from word_forge.relationships import (
    RELATIONSHIP_TYPES,
    get_relationship_properties,
    is_bidirectional,
    get_relationship_weight,
    get_relationship_color,
)


class TestRelationshipTypes:
    """Tests for the RELATIONSHIP_TYPES dictionary."""

    def test_relationship_types_is_dict(self):
        """Test that RELATIONSHIP_TYPES is a dictionary."""
        assert isinstance(RELATIONSHIP_TYPES, dict)

    def test_relationship_types_not_empty(self):
        """Test that RELATIONSHIP_TYPES contains entries."""
        assert len(RELATIONSHIP_TYPES) > 0

    def test_core_relationships_exist(self):
        """Test that core relationships are defined."""
        core = ["synonym", "antonym"]
        for rel in core:
            assert rel in RELATIONSHIP_TYPES

    def test_hierarchical_relationships_exist(self):
        """Test that hierarchical relationships are defined."""
        hierarchical = ["hypernym", "hyponym"]
        for rel in hierarchical:
            assert rel in RELATIONSHIP_TYPES

    def test_part_whole_relationships_exist(self):
        """Test that part-whole relationships are defined."""
        part_whole = ["holonym", "meronym"]
        for rel in part_whole:
            assert rel in RELATIONSHIP_TYPES

    def test_emotional_relationships_exist(self):
        """Test that emotional relationships are defined."""
        emotional = [
            "emotional_synonym",
            "emotional_antonym",
            "intensifies",
            "diminishes",
            "evokes",
            "responds_to",
        ]
        for rel in emotional:
            assert rel in RELATIONSHIP_TYPES

    def test_default_relationship_exists(self):
        """Test that default fallback relationship is defined."""
        assert "default" in RELATIONSHIP_TYPES

    def test_all_relationships_have_required_keys(self):
        """Test that all relationships have weight, color, and bidirectional."""
        required_keys = {"weight", "color", "bidirectional"}
        for rel_type, props in RELATIONSHIP_TYPES.items():
            assert required_keys <= set(props.keys()), f"{rel_type} missing keys"


class TestRelationshipPropertyValues:
    """Tests for the values in relationship properties."""

    def test_weights_in_valid_range(self):
        """Test that all weights are between 0.0 and 1.0."""
        for rel_type, props in RELATIONSHIP_TYPES.items():
            weight = props["weight"]
            assert 0.0 <= weight <= 1.0, f"{rel_type} has invalid weight {weight}"

    def test_colors_are_hex_codes(self):
        """Test that all colors are valid hex color codes."""
        import re

        hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
        for rel_type, props in RELATIONSHIP_TYPES.items():
            color = props["color"]
            assert hex_pattern.match(color), f"{rel_type} has invalid color {color}"

    def test_bidirectional_is_boolean(self):
        """Test that all bidirectional values are booleans."""
        for rel_type, props in RELATIONSHIP_TYPES.items():
            assert isinstance(
                props["bidirectional"], bool
            ), f"{rel_type} bidirectional is not bool"

    def test_synonym_is_bidirectional(self):
        """Test that synonyms are bidirectional."""
        assert RELATIONSHIP_TYPES["synonym"]["bidirectional"] is True

    def test_antonym_is_bidirectional(self):
        """Test that antonyms are bidirectional."""
        assert RELATIONSHIP_TYPES["antonym"]["bidirectional"] is True

    def test_hypernym_is_not_bidirectional(self):
        """Test that hypernyms are not bidirectional."""
        assert RELATIONSHIP_TYPES["hypernym"]["bidirectional"] is False

    def test_hyponym_is_not_bidirectional(self):
        """Test that hyponyms are not bidirectional."""
        assert RELATIONSHIP_TYPES["hyponym"]["bidirectional"] is False

    def test_synonym_has_highest_weight(self):
        """Test that synonyms have the highest weight (1.0)."""
        assert RELATIONSHIP_TYPES["synonym"]["weight"] == 1.0


class TestGetRelationshipProperties:
    """Tests for the get_relationship_properties function."""

    def test_returns_properties_for_known_type(self):
        """Test that function returns correct properties for known type."""
        props = get_relationship_properties("synonym")
        assert props["weight"] == 1.0
        assert props["bidirectional"] is True

    def test_returns_default_for_unknown_type(self):
        """Test that function returns default properties for unknown type."""
        props = get_relationship_properties("unknown_relationship_type")
        default = RELATIONSHIP_TYPES["default"]
        assert props == default

    def test_handles_string_conversion(self):
        """Test that function handles type conversion."""
        props = get_relationship_properties("antonym")
        assert "weight" in props

    def test_returns_relationship_properties_type(self):
        """Test that return value is RelationshipProperties."""
        props = get_relationship_properties("synonym")
        assert "weight" in props
        assert "color" in props
        assert "bidirectional" in props


class TestIsBidirectional:
    """Tests for the is_bidirectional function."""

    def test_synonym_is_bidirectional(self):
        """Test that synonym returns True."""
        assert is_bidirectional("synonym") is True

    def test_antonym_is_bidirectional(self):
        """Test that antonym returns True."""
        assert is_bidirectional("antonym") is True

    def test_hypernym_is_not_bidirectional(self):
        """Test that hypernym returns False."""
        assert is_bidirectional("hypernym") is False

    def test_hyponym_is_not_bidirectional(self):
        """Test that hyponym returns False."""
        assert is_bidirectional("hyponym") is False

    def test_unknown_type_uses_default(self):
        """Test that unknown type uses default bidirectional value."""
        default_bidirectional = RELATIONSHIP_TYPES["default"]["bidirectional"]
        assert is_bidirectional("nonexistent") == default_bidirectional

    def test_emotional_synonym_is_bidirectional(self):
        """Test that emotional_synonym returns True."""
        assert is_bidirectional("emotional_synonym") is True


class TestGetRelationshipWeight:
    """Tests for the get_relationship_weight function."""

    def test_synonym_weight(self):
        """Test synonym weight."""
        assert get_relationship_weight("synonym") == 1.0

    def test_antonym_weight(self):
        """Test antonym weight."""
        assert get_relationship_weight("antonym") == 0.9

    def test_hypernym_weight(self):
        """Test hypernym weight."""
        assert get_relationship_weight("hypernym") == 0.7

    def test_related_weight(self):
        """Test related weight."""
        assert get_relationship_weight("related") == 0.4

    def test_unknown_type_uses_default(self):
        """Test that unknown type uses default weight."""
        default_weight = RELATIONSHIP_TYPES["default"]["weight"]
        assert get_relationship_weight("nonexistent") == default_weight

    def test_returns_float(self):
        """Test that return value is a float."""
        weight = get_relationship_weight("synonym")
        assert isinstance(weight, float)


class TestGetRelationshipColor:
    """Tests for the get_relationship_color function."""

    def test_synonym_color(self):
        """Test synonym color."""
        color = get_relationship_color("synonym")
        assert color == "#4287f5"

    def test_antonym_color(self):
        """Test antonym color."""
        color = get_relationship_color("antonym")
        assert color == "#f54242"

    def test_unknown_type_uses_default(self):
        """Test that unknown type uses default color."""
        default_color = RELATIONSHIP_TYPES["default"]["color"]
        assert get_relationship_color("nonexistent") == default_color

    def test_returns_string(self):
        """Test that return value is a string."""
        color = get_relationship_color("synonym")
        assert isinstance(color, str)

    def test_returns_hex_color(self):
        """Test that return value is a hex color."""
        color = get_relationship_color("synonym")
        assert color.startswith("#")
        assert len(color) == 7


class TestEmotionalRelationshipTypes:
    """Tests specifically for emotional relationship types."""

    def test_emotional_valence_relationships(self):
        """Test emotional valence relationships."""
        assert "emotional_synonym" in RELATIONSHIP_TYPES
        assert "emotional_antonym" in RELATIONSHIP_TYPES

    def test_emotional_intensity_relationships(self):
        """Test emotional intensity relationships."""
        assert "intensifies" in RELATIONSHIP_TYPES
        assert "diminishes" in RELATIONSHIP_TYPES

    def test_emotional_causality_relationships(self):
        """Test emotional causality relationships."""
        assert "evokes" in RELATIONSHIP_TYPES
        assert "responds_to" in RELATIONSHIP_TYPES

    def test_emotional_dimension_relationships(self):
        """Test emotional dimension relationships."""
        assert "valence_related" in RELATIONSHIP_TYPES
        assert "arousal_related" in RELATIONSHIP_TYPES
        assert "dominance_related" in RELATIONSHIP_TYPES

    def test_emotional_complexity_relationships(self):
        """Test emotional complexity relationships."""
        assert "emotional_component" in RELATIONSHIP_TYPES
        assert "emotional_composite" in RELATIONSHIP_TYPES
        assert "emotional_sequence" in RELATIONSHIP_TYPES

    def test_meta_emotional_relationships(self):
        """Test meta-emotional relationships."""
        assert "meta_emotion" in RELATIONSHIP_TYPES
        assert "emotional_regulation" in RELATIONSHIP_TYPES


class TestContextualRelationships:
    """Tests for contextual relationship types."""

    def test_cultural_context_exists(self):
        """Test cultural_context relationship exists."""
        assert "cultural_context" in RELATIONSHIP_TYPES

    def test_situational_context_exists(self):
        """Test situational_context relationship exists."""
        assert "situational_context" in RELATIONSHIP_TYPES

    def test_temporal_context_exists(self):
        """Test temporal_context relationship exists."""
        assert "temporal_context" in RELATIONSHIP_TYPES

    def test_contextual_relationships_not_bidirectional(self):
        """Test that contextual relationships are not bidirectional."""
        contextual = ["cultural_context", "situational_context", "temporal_context"]
        for rel in contextual:
            assert RELATIONSHIP_TYPES[rel]["bidirectional"] is False


class TestDerivationalRelationships:
    """Tests for derivational relationship types."""

    def test_derived_from_exists(self):
        """Test derived_from relationship exists."""
        assert "derived_from" in RELATIONSHIP_TYPES

    def test_etymological_source_exists(self):
        """Test etymological_source relationship exists."""
        assert "etymological_source" in RELATIONSHIP_TYPES

    def test_derivational_not_bidirectional(self):
        """Test that derivational relationships are not bidirectional."""
        assert RELATIONSHIP_TYPES["derived_from"]["bidirectional"] is False
        assert RELATIONSHIP_TYPES["etymological_source"]["bidirectional"] is False
