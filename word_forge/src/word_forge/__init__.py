"""
Word Forge - Living Lexicon Builder.

Word Forge builds semantic/lexical graphs from natural language input,
creating a living dictionary of terms, relationships, and affective meaning.
"""

from word_forge.forge import __version__

# Core imports
from word_forge.config import Config

# Core functions
from word_forge.parser.lexical_functions import (
    create_lexical_dataset,
    get_wordnet_data,
    get_synsets,
)

# Graph components
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_builder import GraphBuilder
from word_forge.graph.graph_query import GraphQuery

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
