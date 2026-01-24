#!/usr/bin/env python3
"""
Code Forge CLI - Command-line interface for code analysis and indexing.

Standalone Usage:
    code-forge status              # Show code index status
    code-forge analyze <file>      # Analyze a Python file
    code-forge index <path>        # Index codebase
    code-forge search <query>      # Search code elements
    code-forge ingest <file>       # Ingest into code library

Enhanced with other forges:
    - knowledge_forge: Semantic code understanding
    - llm_forge: AI-powered code analysis
"""
from __future__ import annotations
from eidosian_core import eidosian

import sys
from pathlib import Path
from typing import Optional

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import StandardCLI, CommandResult, ForgeDetector

from code_forge import CodeAnalyzer, CodeIndexer, CodeLibrarian, index_forge_codebase

# Default paths
DEFAULT_INDEX_PATH = Path("/home/lloyd/eidosian_forge/data/code_index.json")
DEFAULT_LIBRARY_PATH = Path("/home/lloyd/eidosian_forge/data/code_library.json")


class CodeForgeCLI(StandardCLI):
    """CLI for Code Forge - code analysis and indexing."""
    
    name = "code_forge"
    description = "AST-based code analysis, indexing, and library management"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._analyzer: Optional[CodeAnalyzer] = None
        self._indexer: Optional[CodeIndexer] = None
        self._librarian: Optional[CodeLibrarian] = None
    
    @property
    def analyzer(self) -> CodeAnalyzer:
        """Lazy-load code analyzer."""
        if self._analyzer is None:
            self._analyzer = CodeAnalyzer()
        return self._analyzer
    
    @property
    def indexer(self) -> CodeIndexer:
        """Lazy-load code indexer."""
        if self._indexer is None:
            self._indexer = CodeIndexer(DEFAULT_INDEX_PATH)
        return self._indexer
    
    @property
    def librarian(self) -> CodeLibrarian:
        """Lazy-load code librarian."""
        if self._librarian is None:
            self._librarian = CodeLibrarian(DEFAULT_LIBRARY_PATH)
        return self._librarian
    
    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register code-forge specific commands."""
        
        # Analyze command
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Analyze a Python file",
        )
        analyze_parser.add_argument(
            "path",
            help="Path to Python file",
        )
        analyze_parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Show detailed analysis",
        )
        analyze_parser.set_defaults(func=self._cmd_analyze)
        
        # Index command
        index_parser = subparsers.add_parser(
            "index",
            help="Index a codebase",
        )
        index_parser.add_argument(
            "path",
            nargs="?",
            default="/home/lloyd/eidosian_forge",
            help="Path to index (default: eidosian_forge)",
        )
        index_parser.add_argument(
            "--full",
            action="store_true",
            help="Force full reindex",
        )
        index_parser.set_defaults(func=self._cmd_index)
        
        # Search command
        search_parser = subparsers.add_parser(
            "search",
            help="Search code elements",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )
        search_parser.add_argument(
            "-t", "--type",
            choices=["function", "class", "method", "module"],
            help="Filter by element type",
        )
        search_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=10,
            help="Maximum results (default: 10)",
        )
        search_parser.set_defaults(func=self._cmd_search)
        
        # Ingest command
        ingest_parser = subparsers.add_parser(
            "ingest",
            help="Ingest file into code library",
        )
        ingest_parser.add_argument(
            "path",
            help="Path to file",
        )
        ingest_parser.set_defaults(func=self._cmd_ingest)
        
        # Library command
        lib_parser = subparsers.add_parser(
            "library",
            help="List code library contents",
        )
        lib_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=20,
            help="Maximum items (default: 20)",
        )
        lib_parser.set_defaults(func=self._cmd_library)
        
        # Stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="Show detailed statistics",
        )
        stats_parser.set_defaults(func=self._cmd_stats)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show code forge status."""
        try:
            # Check index
            index_exists = DEFAULT_INDEX_PATH.exists()
            element_count = len(self.indexer.elements) if index_exists else 0
            
            # Count by type
            type_counts = {}
            for elem in self.indexer.elements.values():
                t = elem.element_type
                type_counts[t] = type_counts.get(t, 0) + 1
            
            # Check library
            lib_exists = DEFAULT_LIBRARY_PATH.exists()
            
            integrations = []
            if ForgeDetector.is_available("knowledge_forge"):
                integrations.append("knowledge_forge")
            if ForgeDetector.is_available("llm_forge"):
                integrations.append("llm_forge")
            
            return CommandResult(
                True,
                f"Code Forge operational - {element_count} elements indexed",
                {
                    "elements_indexed": element_count,
                    "types": type_counts,
                    "index_path": str(DEFAULT_INDEX_PATH),
                    "index_exists": index_exists,
                    "library_exists": lib_exists,
                    "integrations": integrations,
                }
            )
        except Exception as e:
            return CommandResult(False, f"Error: {e}")
    
    def _cmd_analyze(self, args) -> None:
        """Analyze a Python file."""
        try:
            path = Path(args.path)
            if not path.exists():
                result = CommandResult(False, f"File not found: {path}")
            elif not path.is_file():
                result = CommandResult(False, "Path must be a file")
            else:
                analysis = self.analyzer.analyze_file(path)
                
                if args.verbose:
                    result = CommandResult(
                        True,
                        f"Analyzed {path.name}",
                        analysis
                    )
                else:
                    summary = {
                        "file": str(path),
                        "functions": len(analysis.get("functions", [])),
                        "classes": len(analysis.get("classes", [])),
                        "imports": len(analysis.get("imports", [])),
                    }
                    result = CommandResult(
                        True,
                        f"{path.name}: {summary['functions']} functions, {summary['classes']} classes",
                        summary
                    )
        except Exception as e:
            result = CommandResult(False, f"Analysis error: {e}")
        self._output(result, args)
    
    def _cmd_index(self, args) -> None:
        """Index a codebase."""
        try:
            path = Path(args.path)
            if not path.exists():
                result = CommandResult(False, f"Path not found: {path}")
            else:
                count = self.indexer.index_directory(path)
                result = CommandResult(
                    True,
                    f"Indexed {count} elements from {path}",
                    {
                        "path": str(path),
                        "elements_indexed": count,
                        "total_elements": len(self.indexer.elements),
                    }
                )
        except Exception as e:
            result = CommandResult(False, f"Indexing error: {e}")
        self._output(result, args)
    
    def _cmd_search(self, args) -> None:
        """Search code elements."""
        try:
            results = []
            query_lower = args.query.lower()
            
            for elem in self.indexer.elements.values():
                # Filter by type if specified
                if args.type and elem.element_type != args.type:
                    continue
                
                # Match by name or docstring
                if (query_lower in elem.name.lower() or 
                    query_lower in elem.qualified_name.lower() or
                    (elem.docstring and query_lower in elem.docstring.lower())):
                    results.append({
                        "name": elem.name,
                        "type": elem.element_type,
                        "qualified_name": elem.qualified_name,
                        "file": elem.file_path,
                        "line": elem.line_start,
                    })
                    if len(results) >= args.limit:
                        break
            
            result = CommandResult(
                True,
                f"Found {len(results)} matches for '{args.query}'",
                {"results": results, "query": args.query}
            )
        except Exception as e:
            result = CommandResult(False, f"Search error: {e}")
        self._output(result, args)
    
    def _cmd_ingest(self, args) -> None:
        """Ingest file into code library."""
        try:
            path = Path(args.path)
            if not path.is_file():
                result = CommandResult(False, "Path must be a file")
            else:
                analysis = self.analyzer.analyze_file(path)
                content = path.read_text(encoding="utf-8")
                
                sid = self.librarian.add_snippet(content, metadata=analysis)
                result = CommandResult(
                    True,
                    f"Ingested {path.name} as {sid[:8]}",
                    {"snippet_id": sid, "file": str(path)}
                )
        except Exception as e:
            result = CommandResult(False, f"Ingest error: {e}")
        self._output(result, args)
    
    def _cmd_library(self, args) -> None:
        """List code library contents."""
        try:
            snippets = list(self.librarian.snippets.items())[:args.limit]
            
            items = []
            for sid, snippet in snippets:
                items.append({
                    "id": sid[:8],
                    "preview": snippet.get("content", "")[:60] + "...",
                })
            
            result = CommandResult(
                True,
                f"Library contains {len(self.librarian.snippets)} snippets",
                {"snippets": items, "total": len(self.librarian.snippets)}
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)
    
    def _cmd_stats(self, args) -> None:
        """Show detailed statistics."""
        try:
            # Element types
            type_counts = {}
            file_counts = {}
            
            for elem in self.indexer.elements.values():
                t = elem.element_type
                type_counts[t] = type_counts.get(t, 0) + 1
                
                # Count files
                f = Path(elem.file_path).name
                file_counts[f] = file_counts.get(f, 0) + 1
            
            # Top files
            top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            data = {
                "total_elements": len(self.indexer.elements),
                "by_type": type_counts,
                "unique_files": len(file_counts),
                "top_files": [{"file": f, "elements": c} for f, c in top_files],
                "library_size": len(self.librarian.snippets),
            }
            
            result = CommandResult(
                True,
                f"Stats: {data['total_elements']} elements, {data['unique_files']} files, {data['library_size']} snippets",
                data
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)


@eidosian()
def main():
    """Entry point for code-forge CLI."""
    cli = CodeForgeCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
