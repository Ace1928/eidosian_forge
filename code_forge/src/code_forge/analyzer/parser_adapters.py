from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol


class ParserAdapter(Protocol):
    """Adapter contract for language-aware structural analysis."""

    def supports_language(self, language: str) -> bool:
        ...

    def analyze_file(self, file_path: Path, source: str, language: str) -> Optional[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class _CapturedNode:
    unit_type: str
    name: str
    line_start: int
    line_end: int
    col_start: int
    col_end: int
    source: str


class TreeSitterAdapter:
    """
    Optional tree-sitter-backed adapter for deeper non-Python parsing.

    Falls back to `None` when parser dependencies are unavailable.
    """

    _LANGUAGE_MAP = {
        "javascript": "javascript",
        "typescript": "typescript",
        "tsx": "tsx",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "ruby": "ruby",
        "php": "php",
        "c": "c",
        "cpp": "cpp",
        "csharp": "c_sharp",
        "kotlin": "kotlin",
        "swift": "swift",
        "scala": "scala",
        "lua": "lua",
        "html": "html",
        "css": "css",
        "json": "json",
        "yaml": "yaml",
        "toml": "toml",
    }

    _TYPE_RULES: dict[str, dict[str, set[str]]] = {
        "javascript": {
            "class": {"class_declaration"},
            "function": {"function_declaration", "generator_function_declaration"},
            "method": {"method_definition"},
            "import": {"import_statement", "lexical_declaration"},
        },
        "typescript": {
            "class": {"class_declaration", "interface_declaration"},
            "function": {"function_declaration"},
            "method": {"method_definition", "method_signature"},
            "import": {"import_statement"},
        },
        "go": {
            "class": {"type_declaration"},
            "function": {"function_declaration", "method_declaration"},
            "import": {"import_declaration"},
        },
        "rust": {
            "class": {"struct_item", "enum_item", "trait_item", "impl_item"},
            "function": {"function_item"},
            "import": {"use_declaration"},
        },
        "java": {
            "class": {"class_declaration", "interface_declaration", "enum_declaration"},
            "function": {"constructor_declaration", "method_declaration"},
            "import": {"import_declaration"},
        },
        "c": {
            "function": {"function_definition"},
            "import": {"preproc_include"},
        },
        "cpp": {
            "class": {"class_specifier", "struct_specifier"},
            "function": {"function_definition"},
            "import": {"preproc_include", "using_declaration"},
        },
        "csharp": {
            "class": {"class_declaration", "interface_declaration", "struct_declaration"},
            "function": {"method_declaration", "constructor_declaration"},
            "import": {"using_directive"},
        },
    }

    _CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]{1,120})\s*\(")

    def __init__(self) -> None:
        self._get_parser = self._load_get_parser()
        self._parser_cache: dict[str, Any] = {}

    @staticmethod
    def _load_get_parser():
        try:
            from tree_sitter_language_pack import get_parser  # type: ignore

            return get_parser
        except Exception:
            pass
        try:
            from tree_sitter_languages import get_parser  # type: ignore

            return get_parser
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._get_parser is not None

    def supports_language(self, language: str) -> bool:
        return self.available and (language in self._LANGUAGE_MAP)

    def _resolve_parser(self, language: str):
        if not self.supports_language(language):
            return None
        if language in self._parser_cache:
            return self._parser_cache[language]
        lang_key = self._LANGUAGE_MAP[language]
        try:
            parser = self._get_parser(lang_key) if self._get_parser else None
        except Exception:
            parser = None
        if parser is not None:
            self._parser_cache[language] = parser
        return parser

    @staticmethod
    def _decode(source_bytes: bytes, start: int, end: int) -> str:
        if start < 0 or end < 0 or end <= start:
            return ""
        return source_bytes[start:end].decode("utf-8", errors="ignore")

    @staticmethod
    def _node_name(node: Any, source_bytes: bytes) -> str:
        for field in ("name", "declarator", "property", "field", "identifier"):
            child = node.child_by_field_name(field)
            if child is None:
                continue
            text = TreeSitterAdapter._decode(source_bytes, child.start_byte, child.end_byte).strip()
            if text:
                return text.split(".")[-1]
        for child in getattr(node, "children", [])[:6]:
            ctype = str(getattr(child, "type", ""))
            if ctype.endswith("identifier") or ctype == "identifier":
                text = TreeSitterAdapter._decode(source_bytes, child.start_byte, child.end_byte).strip()
                if text:
                    return text.split(".")[-1]
        snippet = TreeSitterAdapter._decode(source_bytes, node.start_byte, node.end_byte).strip()
        if not snippet:
            return "anonymous"
        head = snippet.splitlines()[0][:120]
        token = re.sub(r"[^A-Za-z0-9_]+", "_", head).strip("_")
        return token[:64] or "anonymous"

    def _capture_nodes(self, language: str, root_node: Any, source_bytes: bytes) -> tuple[list[_CapturedNode], set[str]]:
        rules = self._TYPE_RULES.get(language, {})
        class_types = rules.get("class", set())
        function_types = rules.get("function", set())
        method_types = rules.get("method", set())
        import_types = rules.get("import", set())
        imports: set[str] = set()
        captured: list[_CapturedNode] = []

        def walk(node: Any) -> None:
            ntype = str(getattr(node, "type", ""))
            if ntype in import_types:
                text = self._decode(source_bytes, node.start_byte, node.end_byte).strip()
                if text:
                    imports.add(text[:240])
            if ntype in class_types or ntype in function_types or ntype in method_types:
                unit_type = "class" if ntype in class_types else "function"
                if ntype in method_types:
                    unit_type = "method"
                name = self._node_name(node, source_bytes)
                source = self._decode(source_bytes, node.start_byte, node.end_byte)
                captured.append(
                    _CapturedNode(
                        unit_type=unit_type,
                        name=name,
                        line_start=int(node.start_point[0]) + 1,
                        line_end=int(node.end_point[0]) + 1,
                        col_start=int(node.start_point[1]),
                        col_end=int(node.end_point[1]),
                        source=source,
                    )
                )
            for child in getattr(node, "children", []):
                walk(child)

        walk(root_node)
        return captured, imports

    def _extract_edges(self, module_name: str, nodes: Iterable[_CapturedNode]) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for node in nodes:
            source_qn = f"{module_name}.{node.name}"
            for match in self._CALL_RE.findall(node.source or ""):
                if match in {"if", "for", "while", "switch", "catch", "return"}:
                    continue
                key = ("calls", source_qn, match)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "rel_type": "calls",
                        "source_qualified_name": source_qn,
                        "target": match,
                        "confidence": 0.6,
                    }
                )
        return edges

    def analyze_file(self, file_path: Path, source: str, language: str) -> Optional[dict[str, Any]]:
        parser = self._resolve_parser(language)
        if parser is None:
            return None
        source_bytes = source.encode("utf-8")
        try:
            tree = parser.parse(source_bytes)
        except Exception:
            return None
        root = getattr(tree, "root_node", None)
        if root is None:
            return None

        module_name = file_path.stem
        captured, imports = self._capture_nodes(language, root, source_bytes)
        nodes: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        functions: list[dict[str, Any]] = []
        for rec in captured:
            qualified = f"{module_name}.{rec.name}"
            node = {
                "unit_type": rec.unit_type,
                "name": rec.name,
                "qualified_name": qualified,
                "parent_qualified_name": module_name if rec.unit_type == "class" else None,
                "docstring": None,
                "args": [],
                "line_start": rec.line_start,
                "line_end": rec.line_end,
                "col_start": rec.col_start,
                "col_end": rec.col_end,
                "source": rec.source.strip(),
                "complexity": None,
            }
            nodes.append(node)
            entry = {
                "name": rec.name,
                "docstring": None,
                "source": rec.source.strip(),
                "args": [],
                "line_start": rec.line_start,
                "line_end": rec.line_end,
                "col_start": rec.col_start,
                "col_end": rec.col_end,
            }
            if rec.unit_type == "class":
                entry["methods"] = []
                classes.append(entry)
            else:
                functions.append(entry)

        return {
            "language": language,
            "classes": classes,
            "functions": functions,
            "imports": sorted(imports),
            "docstring": None,
            "nodes": nodes,
            "edges": self._extract_edges(module_name, captured),
            "module": {
                "docstring": None,
                "source": source,
                "line_start": 1,
                "line_end": max(1, len(source.splitlines())),
                "col_start": 0,
                "col_end": 0,
            },
            "parser_adapter": "tree_sitter",
        }


def build_default_parser_adapters() -> list[ParserAdapter]:
    adapter = TreeSitterAdapter()
    if adapter.available:
        return [adapter]
    return []

