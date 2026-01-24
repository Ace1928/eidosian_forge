#!/usr/bin/env python3
"""
Word Forge CLI - Command-line interface for the living lexicon system.

Standalone Usage:
    word-forge status              # Show lexicon status
    word-forge lookup <word>       # Look up word in lexicon
    word-forge define <word>       # Get word definition
    word-forge related <word>      # Find related words
    word-forge graph stats         # Show graph statistics

Enhanced with other forges:
    - knowledge_forge: Cross-link lexical concepts
    - llm_forge: AI-enhanced definitions
"""
from __future__ import annotations
from eidosian_core import eidosian

import sys
from pathlib import Path
from typing import Optional

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import StandardCLI, CommandResult, ForgeDetector


class WordForgeCLI(StandardCLI):
    """CLI for Word Forge - living lexicon system."""
    
    name = "word_forge"
    description = "Living lexicon with semantic graphs and affective understanding"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._config = None
        self._graph_manager = None
        self._db_manager = None
    
    @property
    def config(self):
        """Lazy-load config."""
        if self._config is None:
            from word_forge.config import Config
            self._config = Config()
        return self._config
    
    @property
    def graph_manager(self):
        """Lazy-load graph manager."""
        if self._graph_manager is None:
            try:
                from word_forge.graph.graph_manager import GraphManager
                self._graph_manager = GraphManager()
            except Exception as e:
                # GraphManager may take time to initialize
                return None
        return self._graph_manager
    
    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register word-forge specific commands."""
        
        # Lookup command
        lookup_parser = subparsers.add_parser(
            "lookup",
            help="Look up a word in the lexicon",
        )
        lookup_parser.add_argument(
            "word",
            help="Word to look up",
        )
        lookup_parser.set_defaults(func=self._cmd_lookup)
        
        # Define command
        define_parser = subparsers.add_parser(
            "define",
            help="Get word definition from WordNet",
        )
        define_parser.add_argument(
            "word",
            help="Word to define",
        )
        define_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=3,
            help="Maximum definitions (default: 3)",
        )
        define_parser.set_defaults(func=self._cmd_define)
        
        # Related command
        related_parser = subparsers.add_parser(
            "related",
            help="Find related words",
        )
        related_parser.add_argument(
            "word",
            help="Word to find relations for",
        )
        related_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=10,
            help="Maximum results (default: 10)",
        )
        related_parser.set_defaults(func=self._cmd_related)
        
        # Synsets command
        synsets_parser = subparsers.add_parser(
            "synsets",
            help="Get WordNet synsets for a word",
        )
        synsets_parser.add_argument(
            "word",
            help="Word to get synsets for",
        )
        synsets_parser.set_defaults(func=self._cmd_synsets)
        
        # Graph stats command
        graph_parser = subparsers.add_parser(
            "graph",
            help="Show graph statistics",
        )
        graph_parser.set_defaults(func=self._cmd_graph)
        
        # Build command
        build_parser = subparsers.add_parser(
            "build",
            help="Build lexical entry for a word",
        )
        build_parser.add_argument(
            "word",
            help="Word to build entry for",
        )
        build_parser.set_defaults(func=self._cmd_build)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show word forge status."""
        try:
            from word_forge import __version__
            
            # Check components
            components = {
                "config": False,
                "wordnet": False,
                "graph": False,
            }
            
            # Check config
            try:
                from word_forge.config import Config
                config = Config()
                components["config"] = True
            except Exception:
                pass
            
            # Check WordNet
            try:
                from nltk.corpus import wordnet
                test = wordnet.synsets("test")
                components["wordnet"] = len(test) > 0
            except Exception:
                pass
            
            # Check graph (may be slow)
            try:
                from word_forge.graph.graph_builder import GraphBuilder
                components["graph"] = True
            except Exception:
                pass
            
            integrations = []
            if ForgeDetector.is_available("knowledge_forge"):
                integrations.append("knowledge_forge")
            if ForgeDetector.is_available("llm_forge"):
                integrations.append("llm_forge")
            
            all_ok = all(components.values())
            
            return CommandResult(
                True,
                f"Word Forge {'operational' if all_ok else 'partial'} - v{__version__}",
                {
                    "version": __version__,
                    "components": components,
                    "integrations": integrations,
                }
            )
        except Exception as e:
            return CommandResult(False, f"Error: {e}")
    
    def _cmd_lookup(self, args) -> None:
        """Look up a word in the lexicon."""
        try:
            from word_forge.parser.lexical_functions import get_wordnet_data
            
            data = get_wordnet_data(args.word)
            
            if not data:
                result = CommandResult(
                    True,
                    f"No data found for '{args.word}'",
                    {"word": args.word, "found": False}
                )
            else:
                result = CommandResult(
                    True,
                    f"Found data for '{args.word}'",
                    {
                        "word": args.word,
                        "found": True,
                        "data_keys": list(data.keys()) if isinstance(data, dict) else ["raw"],
                    }
                )
        except Exception as e:
            result = CommandResult(False, f"Lookup error: {e}")
        self._output(result, args)
    
    def _cmd_define(self, args) -> None:
        """Get word definition from WordNet."""
        try:
            from nltk.corpus import wordnet
            
            synsets = wordnet.synsets(args.word)
            
            if not synsets:
                result = CommandResult(
                    True,
                    f"No definitions found for '{args.word}'",
                    {"word": args.word, "definitions": []}
                )
            else:
                definitions = []
                for syn in synsets[:args.limit]:
                    definitions.append({
                        "name": syn.name(),
                        "pos": syn.pos(),
                        "definition": syn.definition(),
                        "examples": syn.examples()[:2],
                    })
                
                result = CommandResult(
                    True,
                    f"Found {len(synsets)} definitions for '{args.word}'",
                    {
                        "word": args.word,
                        "total_synsets": len(synsets),
                        "definitions": definitions,
                    }
                )
        except Exception as e:
            result = CommandResult(False, f"Define error: {e}")
        self._output(result, args)
    
    def _cmd_related(self, args) -> None:
        """Find related words."""
        try:
            from nltk.corpus import wordnet
            
            synsets = wordnet.synsets(args.word)
            
            if not synsets:
                result = CommandResult(
                    True,
                    f"No synsets found for '{args.word}'",
                    {"word": args.word, "related": []}
                )
            else:
                related = set()
                for syn in synsets[:3]:
                    # Hypernyms (more general)
                    for hyper in syn.hypernyms():
                        for lemma in hyper.lemmas():
                            related.add(("hypernym", lemma.name().replace("_", " ")))
                    
                    # Hyponyms (more specific)
                    for hypo in syn.hyponyms():
                        for lemma in hypo.lemmas():
                            related.add(("hyponym", lemma.name().replace("_", " ")))
                    
                    # Synonyms
                    for lemma in syn.lemmas():
                        if lemma.name().lower() != args.word.lower():
                            related.add(("synonym", lemma.name().replace("_", " ")))
                
                related_list = [{"type": t, "word": w} for t, w in list(related)[:args.limit]]
                
                result = CommandResult(
                    True,
                    f"Found {len(related_list)} related words for '{args.word}'",
                    {
                        "word": args.word,
                        "related": related_list,
                    }
                )
        except Exception as e:
            result = CommandResult(False, f"Related error: {e}")
        self._output(result, args)
    
    def _cmd_synsets(self, args) -> None:
        """Get WordNet synsets for a word."""
        try:
            from word_forge.parser.lexical_functions import get_synsets
            
            synsets = get_synsets(args.word)
            
            synset_data = []
            for syn in synsets[:10]:
                synset_data.append({
                    "name": syn.name(),
                    "pos": syn.pos(),
                    "definition": syn.definition()[:100],
                })
            
            result = CommandResult(
                True,
                f"Found {len(synsets)} synsets for '{args.word}'",
                {
                    "word": args.word,
                    "count": len(synsets),
                    "synsets": synset_data,
                }
            )
        except Exception as e:
            result = CommandResult(False, f"Synsets error: {e}")
        self._output(result, args)
    
    def _cmd_graph(self, args) -> None:
        """Show graph statistics."""
        try:
            # Try to get graph stats without full initialization
            from word_forge.config import config
            
            db_path = config.db_path
            graph_path = config.data_dir / "graph.pkl"
            
            data = {
                "db_path": str(db_path),
                "db_exists": db_path.exists() if db_path else False,
                "graph_path": str(graph_path),
                "graph_exists": graph_path.exists() if graph_path else False,
            }
            
            # Try to get node count from DB
            if db_path and db_path.exists():
                try:
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.execute("SELECT COUNT(*) FROM words")
                    data["word_count"] = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    data["word_count"] = "unknown"
            
            result = CommandResult(
                True,
                f"Graph stats: {data.get('word_count', 'N/A')} words in database",
                data
            )
        except Exception as e:
            result = CommandResult(False, f"Graph error: {e}")
        self._output(result, args)
    
    def _cmd_build(self, args) -> None:
        """Build lexical entry for a word."""
        try:
            from word_forge.parser.lexical_functions import create_lexical_dataset
            
            data = create_lexical_dataset(args.word)
            
            if not data:
                result = CommandResult(
                    True,
                    f"No lexical data generated for '{args.word}'",
                    {"word": args.word, "built": False}
                )
            else:
                result = CommandResult(
                    True,
                    f"Built lexical entry for '{args.word}'",
                    {
                        "word": args.word,
                        "built": True,
                        "data_keys": list(data.keys()) if isinstance(data, dict) else ["raw"],
                    }
                )
        except Exception as e:
            result = CommandResult(False, f"Build error: {e}")
        self._output(result, args)


@eidosian()
def main():
    """Entry point for word-forge CLI."""
    cli = WordForgeCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
