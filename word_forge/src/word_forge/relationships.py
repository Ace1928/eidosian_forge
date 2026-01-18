"""
Word relationship type definitions for the Word Forge lexical network.

This module defines semantic and emotional relationship types used throughout
the application to represent connections between lexical items. Each relationship
has properties defining its weight (importance), visual representation, and
directionality characteristics.

The relationships are organized into semantic categories with consistent
property patterns for each relationship type.

Typical usage:
    from word_forge.relationships import RELATIONSHIP_TYPES, get_relationship_properties

    # Get properties for a specific relationship
    synonym_props = RELATIONSHIP_TYPES["synonym"]
    weight = synonym_props["weight"]  # 1.0

    # Check if a relationship is bidirectional
    is_bidirectional = RELATIONSHIP_TYPES["hypernym"]["bidirectional"]  # False

    # Use the helper function with fallback to default
    props = get_relationship_properties("unknown_type")  # Returns default properties
"""

from typing import Dict, Final, Literal, TypedDict, Union


class RelationshipProperties(TypedDict):
    """Properties defining a lexical or emotional relationship type.

    Each relationship has a set of properties that define its characteristics,
    visual representation, and behavioral aspects in the lexical network.

    Attributes:
        weight: Float between 0.0-1.0 indicating relationship strength/importance.
            Higher values represent stronger relationships (1.0 = strongest).
        color: Hexadecimal color code for visual representation in UI.
        bidirectional: Whether the relationship applies equally in both directions.
            If True, the relationship is symmetric (A→B implies B→A).
            If False, the relationship is directional (A→B does not imply B→A).
    """

    weight: float
    color: str
    bidirectional: bool


# Type definitions for relationship categories
# Each category represents a distinct semantic or emotional dimension

#: Core linguistic relationships between words (strongest connections)
CORE_RELATIONSHIPS = Literal["synonym", "antonym"]

#: Relationships representing taxonomic hierarchies (superordinate/subordinate)
HIERARCHICAL_RELATIONSHIPS = Literal["hypernym", "hyponym"]

#: Relationships representing compositional hierarchies (whole/part)
PART_WHOLE_RELATIONSHIPS = Literal["holonym", "meronym"]

#: Cross-language equivalence relationships
TRANSLATION_RELATIONSHIPS = Literal["translation"]

#: Relationships connecting words to their semantic domains or functions
SEMANTIC_FIELD_RELATIONSHIPS = Literal["domain", "function"]

#: General semantic association without specific relationship type
GENERAL_SEMANTIC_RELATIONSHIPS = Literal["related"]

#: Relationships based on word formation and etymology
DERIVATIONAL_RELATIONSHIPS = Literal["derived_from", "etymological_source"]

#: Relationships defining usage contexts and registers
USAGE_RELATIONSHIPS = Literal["context", "register"]

#: Relationships connecting concepts to their examples or instances
EXAMPLE_BASED_RELATIONSHIPS = Literal["example_of", "instance"]

#: Relationships between words with similar or opposite emotional content
EMOTIONAL_VALENCE_RELATIONSHIPS = Literal["emotional_synonym", "emotional_antonym"]

#: Relationships that modify emotional intensity
EMOTIONAL_INTENSITY_RELATIONSHIPS = Literal["intensifies", "diminishes"]

#: Relationships describing emotional cause-effect connections
EMOTIONAL_CAUSALITY_RELATIONSHIPS = Literal["evokes", "responds_to"]

#: Relationships connecting words along specific emotional dimensions
EMOTIONAL_DIMENSION_RELATIONSHIPS = Literal[
    "valence_related", "arousal_related", "dominance_related"
]

#: Relationships describing compositional aspects of complex emotions
EMOTIONAL_COMPLEXITY_RELATIONSHIPS = Literal[
    "emotional_component", "emotional_composite", "emotional_sequence"
]

#: Relationships defining how emotions relate to different contexts
CONTEXTUAL_EMOTIONAL_RELATIONSHIPS = Literal[
    "cultural_context", "situational_context", "temporal_context"
]

#: Relationships describing higher-order emotional processing
META_EMOTIONAL_RELATIONSHIPS = Literal["meta_emotion", "emotional_regulation"]

#: Default relationship type for fallback behavior
DEFAULT_RELATIONSHIP = Literal["default"]

# Combined type for all relationship types
RelationshipType = Union[
    CORE_RELATIONSHIPS,
    HIERARCHICAL_RELATIONSHIPS,
    PART_WHOLE_RELATIONSHIPS,
    TRANSLATION_RELATIONSHIPS,
    SEMANTIC_FIELD_RELATIONSHIPS,
    GENERAL_SEMANTIC_RELATIONSHIPS,
    DERIVATIONAL_RELATIONSHIPS,
    USAGE_RELATIONSHIPS,
    EXAMPLE_BASED_RELATIONSHIPS,
    EMOTIONAL_VALENCE_RELATIONSHIPS,
    EMOTIONAL_INTENSITY_RELATIONSHIPS,
    EMOTIONAL_CAUSALITY_RELATIONSHIPS,
    EMOTIONAL_DIMENSION_RELATIONSHIPS,
    EMOTIONAL_COMPLEXITY_RELATIONSHIPS,
    CONTEXTUAL_EMOTIONAL_RELATIONSHIPS,
    META_EMOTIONAL_RELATIONSHIPS,
    DEFAULT_RELATIONSHIP,
    str,  # Allow string fallback for backward compatibility
]


# Expanded relationship types to include all relationships from ParserRefiner
RELATIONSHIP_TYPES: Final[Dict[str, RelationshipProperties]] = {
    # Core relationships
    "synonym": {"weight": 1.0, "color": "#4287f5", "bidirectional": True},
    "antonym": {"weight": 0.9, "color": "#f54242", "bidirectional": True},
    # Hierarchical relationships
    "hypernym": {"weight": 0.7, "color": "#42f584", "bidirectional": False},
    "hyponym": {"weight": 0.7, "color": "#a142f5", "bidirectional": False},
    # Part-whole relationships
    "holonym": {"weight": 0.6, "color": "#f5a142", "bidirectional": False},
    "meronym": {"weight": 0.6, "color": "#42f5f5", "bidirectional": False},
    # Translation relationships
    "translation": {"weight": 0.8, "color": "#42d4f5", "bidirectional": True},
    # Semantic field relationships
    "domain": {"weight": 0.5, "color": "#7a42f5", "bidirectional": False},
    "function": {"weight": 0.5, "color": "#f542a7", "bidirectional": False},
    # General semantic relationships
    "related": {"weight": 0.4, "color": "#42f5a1", "bidirectional": True},
    # Derivational relationships
    "derived_from": {"weight": 0.5, "color": "#8c42f5", "bidirectional": False},
    "etymological_source": {"weight": 0.4, "color": "#f5b942", "bidirectional": False},
    # Usage relationships
    "context": {"weight": 0.3, "color": "#42d4f5", "bidirectional": False},
    "register": {"weight": 0.3, "color": "#f542d4", "bidirectional": False},
    # Example-based relationships
    "example_of": {"weight": 0.3, "color": "#7adbf5", "bidirectional": False},
    "instance": {"weight": 0.4, "color": "#e642f5", "bidirectional": False},
    # Emotional valence relationships
    "emotional_synonym": {"weight": 0.85, "color": "#8a2be2", "bidirectional": True},
    "emotional_antonym": {"weight": 0.85, "color": "#ff6347", "bidirectional": True},
    # Emotional intensity relationships
    "intensifies": {"weight": 0.70, "color": "#ff4500", "bidirectional": False},
    "diminishes": {"weight": 0.70, "color": "#4682b4", "bidirectional": False},
    # Emotional causality relationships
    "evokes": {"weight": 0.65, "color": "#da70d6", "bidirectional": False},
    "responds_to": {"weight": 0.65, "color": "#32cd32", "bidirectional": False},
    # Emotional dimension relationships
    "valence_related": {"weight": 0.60, "color": "#ff69b4", "bidirectional": True},
    "arousal_related": {"weight": 0.60, "color": "#ff8c00", "bidirectional": True},
    "dominance_related": {"weight": 0.60, "color": "#4b0082", "bidirectional": True},
    # Emotional complexity relationships
    "emotional_component": {"weight": 0.55, "color": "#7b68ee", "bidirectional": False},
    "emotional_composite": {"weight": 0.55, "color": "#ee82ee", "bidirectional": False},
    "emotional_sequence": {"weight": 0.50, "color": "#6a5acd", "bidirectional": False},
    # Contextual emotional relationships
    "cultural_context": {"weight": 0.45, "color": "#daa520", "bidirectional": False},
    "situational_context": {"weight": 0.45, "color": "#20b2aa", "bidirectional": False},
    "temporal_context": {"weight": 0.45, "color": "#778899", "bidirectional": False},
    # Meta-emotional relationships
    "meta_emotion": {"weight": 0.40, "color": "#800080", "bidirectional": False},
    "emotional_regulation": {
        "weight": 0.40,
        "color": "#008080",
        "bidirectional": False,
    },
    # Default for any other relationship
    "default": {"weight": 0.3, "color": "#aaaaaa", "bidirectional": True},
}


def get_relationship_properties(
    relationship_type: RelationshipType,
) -> RelationshipProperties:
    """Get properties for a relationship type with fallback to default.

    This function safely retrieves relationship properties even for undefined
    relationship types, ensuring robust behavior in all contexts.

    Args:
        relationship_type: The relationship type to retrieve properties for.
            Can be any string, but preferably one of the defined relationship types.

    Returns:
        RelationshipProperties for the specified relationship, or default properties
        if the relationship type isn't defined.

    Examples:
        >>> get_relationship_properties("synonym")
        {'weight': 1.0, 'color': '#4287f5', 'bidirectional': True}

        >>> get_relationship_properties("unknown_type")
        {'weight': 0.3, 'color': '#aaaaaa', 'bidirectional': True}
    """
    return RELATIONSHIP_TYPES.get(str(relationship_type), RELATIONSHIP_TYPES["default"])


def is_bidirectional(relationship_type: RelationshipType) -> bool:
    """Check if a relationship type is bidirectional.

    Determines whether a relationship applies equally in both directions
    by retrieving its properties and checking the bidirectional flag.

    Args:
        relationship_type: The relationship type to check.

    Returns:
        True if the relationship is bidirectional, False otherwise.

    Examples:
        >>> is_bidirectional("synonym")
        True

        >>> is_bidirectional("hypernym")
        False
    """
    return get_relationship_properties(relationship_type)["bidirectional"]


def get_relationship_weight(relationship_type: RelationshipType) -> float:
    """Get the weight value for a relationship type.

    Retrieves the importance weight of a relationship type, which indicates
    its strength or significance in the lexical network.

    Args:
        relationship_type: The relationship type to get the weight for.

    Returns:
        Float between 0.0-1.0 representing the relationship's weight.

    Examples:
        >>> get_relationship_weight("synonym")
        1.0

        >>> get_relationship_weight("related")
        0.4
    """
    return get_relationship_properties(relationship_type)["weight"]


def get_relationship_color(relationship_type: RelationshipType) -> str:
    """Get the visual color code for a relationship type.

    Retrieves the hex color code used to visually represent this
    relationship type in UI components.

    Args:
        relationship_type: The relationship type to get the color for.

    Returns:
        Hexadecimal color code as a string (e.g., "#4287f5").

    Examples:
        >>> get_relationship_color("synonym")
        '#4287f5'

        >>> get_relationship_color("antonym")
        '#f54242'
    """
    return get_relationship_properties(relationship_type)["color"]
