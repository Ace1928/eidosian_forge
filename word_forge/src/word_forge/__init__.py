"""
Word Forge - Living Lexicon Builder.

Word Forge builds semantic/lexical graphs from natural language input,
creating a living dictionary of terms, relationships, and affective meaning.
"""

# Core imports
from word_forge.config import Config
from word_forge.forge import __version__
from word_forge.graph.graph_builder import GraphBuilder

# Graph components
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_query import GraphQuery

# Core functions
from word_forge.parser.lexical_functions import (
    create_lexical_dataset,
    get_synsets,
    get_wordnet_data,
)

__all__ = [
    "__version__",
    "Config",
    "create_lexical_dataset",
    "get_wordnet_data",
    "get_synsets",
    "GraphManager",
    "GraphBuilder",
    "GraphQuery",
]
