from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Optional

from eidosian_core import eidosian

from code_forge.analyzer.parser_adapters import ParserAdapter, build_default_parser_adapters

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".kt": "kotlin",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".xml": "xml",
    ".md": "markdown",
}


class GenericCodeAnalyzer:
    """Regex-oriented analyzer for non-Python source files."""

    def __init__(self, adapters: Optional[list[ParserAdapter]] = None) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            ("class", re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")),
            ("interface", re.compile(r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)")),
            ("trait", re.compile(r"\btrait\s+([A-Za-z_][A-Za-z0-9_]*)")),
            ("enum", re.compile(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)")),
            (
                "function",
                re.compile(r"\b(?:function|fn|func)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
            ),
            (
                "function",
                re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"),
            ),
            (
                "function",
                re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\([^)]*\)\s*=>"),
            ),
            (
                "method",
                re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"),
            ),
        ]
        if adapters is not None:
            self._adapters = list(adapters)
        elif os.environ.get("EIDOS_CODE_FORGE_DISABLE_TREE_SITTER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self._adapters = []
        else:
            self._adapters = build_default_parser_adapters()

    @staticmethod
    def detect_language(file_path: Path) -> str:
        return _LANGUAGE_BY_EXT.get(file_path.suffix.lower(), "text")

    @staticmethod
    def supported_extensions() -> set[str]:
        return set(_LANGUAGE_BY_EXT)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _line_source(lines: list[str], idx: int) -> str:
        if idx < 0 or idx >= len(lines):
            return ""
        return lines[idx].rstrip("\n")

    @eidosian()
    def analyze_file(self, file_path: Path) -> dict[str, Any]:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            return {"error": str(exc), "file": str(file_path)}

        lines = source.splitlines()
        module_name = file_path.stem
        language = self.detect_language(file_path)

        for adapter in self._adapters:
            try:
                if adapter.supports_language(language):
                    adapted = adapter.analyze_file(file_path=file_path, source=source, language=language)
                    if isinstance(adapted, dict) and adapted:
                        adapted.setdefault("language", language)
                        return adapted
            except Exception:
                continue

        nodes: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        functions: list[dict[str, Any]] = []
        imports: list[str] = []
        edges: list[dict[str, Any]] = []

        seen_signatures: set[tuple[str, str, int]] = set()
        active_parent: Optional[str] = None
        edge_seen: set[tuple[str, str, str]] = set()

        def add_edge(rel_type: str, source_qn: str, target: str) -> None:
            key = (rel_type, source_qn, target)
            if key in edge_seen:
                return
            edge_seen.add(key)
            edges.append(
                {
                    "rel_type": rel_type,
                    "source_qualified_name": source_qn,
                    "target": target,
                    "confidence": 0.6,
                }
            )

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("//", "#", "*", "/*", "*/", "<!--")):
                continue

            owner_qn = active_parent or "__module__"

            import_match = re.match(r"^(?:import|from)\s+([A-Za-z0-9_./-]+)", stripped)
            if import_match:
                target = import_match.group(1).replace("/", ".")
                imports.append(target)
                add_edge("imports", owner_qn, target)

            require_match = re.findall(r"(?:require|import)\s*\(\s*[\"']([^\"']+)[\"']\s*\)", stripped)
            for target in require_match:
                imports.append(target)
                add_edge("imports", owner_qn, target)

            call_matches = re.findall(r"\b([A-Za-z_][A-Za-z0-9_.]{1,120})\s*\(", stripped)
            for called in call_matches:
                if called in {"if", "for", "while", "switch", "catch", "return"}:
                    continue
                add_edge("calls", owner_qn, called)

            for unit_type, pattern in self._patterns:
                match = pattern.search(stripped)
                if not match:
                    continue

                name = match.group(1)
                signature = (unit_type, name, line_no)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                qualified = f"{module_name}.{name}"
                parent_qn = active_parent
                if unit_type in {"class", "interface", "trait", "enum"}:
                    active_parent = qualified
                    parent_qn = module_name
                elif parent_qn and unit_type == "function":
                    unit_type = "method"

                node = {
                    "unit_type": unit_type,
                    "name": name,
                    "qualified_name": qualified,
                    "parent_qualified_name": parent_qn,
                    "docstring": None,
                    "args": [],
                    "line_start": line_no,
                    "line_end": line_no,
                    "col_start": 0,
                    "col_end": len(stripped),
                    "source": stripped,
                    "content_hash": self._hash_text(stripped),
                    "complexity": None,
                }
                nodes.append(node)

                if unit_type in {"class", "interface", "trait", "enum"}:
                    classes.append(
                        {
                            "name": name,
                            "docstring": None,
                            "source": stripped,
                            "methods": [],
                            "line_start": line_no,
                            "line_end": line_no,
                            "col_start": 0,
                            "col_end": len(stripped),
                            "content_hash": node["content_hash"],
                        }
                    )
                elif unit_type in {"function", "method"}:
                    functions.append(
                        {
                            "name": name,
                            "docstring": None,
                            "source": stripped,
                            "args": [],
                            "line_start": line_no,
                            "line_end": line_no,
                            "col_start": 0,
                            "col_end": len(stripped),
                            "content_hash": node["content_hash"],
                        }
                    )
                break

        return {
            "language": language,
            "classes": classes,
            "functions": functions,
            "imports": sorted(set(imports)),
            "docstring": None,
            "module": {
                "docstring": None,
                "source": source,
                "content_hash": self._hash_text(source),
                "line_start": 1,
                "line_end": len(lines),
                "col_start": 0,
                "col_end": len(lines[-1]) if lines else 0,
            },
            "nodes": nodes,
            "edges": edges,
        }
