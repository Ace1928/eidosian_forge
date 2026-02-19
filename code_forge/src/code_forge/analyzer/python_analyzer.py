from eidosian_core import eidosian
"""
Code Analyzer - AST-based source analysis.
"""
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import builtins

class CodeAnalyzer:
    """Analyzes Python source code using the AST module."""
    
    @eidosian()
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a file and return its structure."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            lines = source.splitlines()
            return self._visit_node(tree, lines, source)
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

    def _node_span(self, node: ast.AST) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        if not hasattr(node, "lineno"):
            return None, None, None, None
        line_start = getattr(node, "lineno", None)
        line_end = getattr(node, "end_lineno", None)
        col_start = getattr(node, "col_offset", None)
        col_end = getattr(node, "end_col_offset", None)
        return line_start, line_end, col_start, col_end

    def _extract_source(
        self,
        lines: List[str],
        line_start: Optional[int],
        line_end: Optional[int],
    ) -> str:
        if line_start is None:
            return ""
        if line_end is None:
            line_end = line_start
        start = max(line_start - 1, 0)
        end = max(line_end, start + 1)
        return "\n".join(lines[start:end])

    def _hash_source(self, source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _visit_node(self, node: ast.AST, lines: List[str], source: str) -> Dict[str, Any]:
        """Recursively visit AST nodes and produce granular node list."""
        summary = {
            "classes": [],
            "functions": [],
            "imports": [],
            "edges": [],
            "docstring": ast.get_docstring(node),
            "module": {
                "docstring": ast.get_docstring(node),
                "source": source,
                "content_hash": self._hash_source(source),
                "line_start": 1,
                "line_end": len(lines),
                "col_start": 0,
                "col_end": len(lines[-1]) if lines else 0,
            },
            "nodes": [],
        }

        parent_map: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        def qualified_name_for(n: ast.AST) -> str:
            parts = []
            cur = n
            while cur in parent_map:
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parts.append(cur.name)
                if isinstance(cur, ast.ClassDef):
                    parts.append(cur.name)
                cur = parent_map[cur]
            return ".".join(reversed(parts))

        def parent_qualified_name_for(n: ast.AST) -> Optional[str]:
            cur = parent_map.get(n)
            while cur is not None:
                if isinstance(cur, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    return qualified_name_for(cur)
                cur = parent_map.get(cur)
            return None

        def complexity_for(n: ast.AST) -> Optional[int]:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return None
            complexity = 1
            for sub in ast.walk(n):
                if isinstance(
                    sub,
                    (
                        ast.If,
                        ast.For,
                        ast.While,
                        ast.Try,
                        ast.With,
                        ast.AsyncWith,
                        ast.AsyncFor,
                        ast.ExceptHandler,
                        ast.BoolOp,
                        ast.IfExp,
                        ast.Match,
                    ),
                ):
                    complexity += 1
            return complexity

        def add_node(n: ast.AST, unit_type: str, name: str) -> None:
            line_start, line_end, col_start, col_end = self._node_span(n)
            node_source = self._extract_source(lines, line_start, line_end)
            docstring = None
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(n)
            summary["nodes"].append(
                {
                    "unit_type": unit_type,
                    "name": name,
                    "qualified_name": qualified_name_for(n),
                    "parent_qualified_name": parent_qualified_name_for(n),
                    "docstring": docstring,
                    "args": [a.arg for a in n.args.args] if isinstance(n, ast.FunctionDef) else [],
                    "line_start": line_start,
                    "line_end": line_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "source": node_source,
                    "content_hash": self._hash_source(node_source) if node_source else None,
                    "complexity": complexity_for(n),
                }
            )

        def owner_qualified_name_for(n: ast.AST) -> str:
            cur = n
            while cur in parent_map:
                cur = parent_map[cur]
                if isinstance(cur, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    owner_qn = qualified_name_for(cur)
                    if owner_qn:
                        return owner_qn
            return "__module__"

        def expr_to_name(expr: ast.AST) -> Optional[str]:
            if isinstance(expr, ast.Name):
                return expr.id
            if isinstance(expr, ast.Attribute):
                parts: list[str] = []
                cur: Optional[ast.AST] = expr
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    return ".".join(reversed(parts))
                return expr.attr
            return None

        edge_set: set[Tuple[str, str, str]] = set()
        use_set: set[Tuple[str, str, str]] = set()
        builtin_names = set(dir(builtins))

        def add_edge(rel_type: str, source_qn: str, target: str) -> None:
            key = (rel_type, source_qn, target)
            if key in edge_set:
                return
            edge_set.add(key)
            summary["edges"].append(
                {
                    "rel_type": rel_type,
                    "source_qualified_name": source_qn,
                    "target": target,
                    "confidence": 1.0,
                }
            )

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                line_start, line_end, col_start, col_end = self._node_span(child)
                cls_source = self._extract_source(lines, line_start, line_end)
                summary["classes"].append(
                    {
                        "name": child.name,
                        "docstring": ast.get_docstring(child),
                        "source": cls_source,
                        "methods": [n.name for n in child.body if isinstance(n, ast.FunctionDef)],
                        "line_start": line_start,
                        "line_end": line_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "content_hash": self._hash_source(cls_source) if cls_source else None,
                    }
                )
            elif isinstance(child, ast.FunctionDef):
                line_start, line_end, col_start, col_end = self._node_span(child)
                func_source = self._extract_source(lines, line_start, line_end)
                summary["functions"].append(
                    {
                        "name": child.name,
                        "docstring": ast.get_docstring(child),
                        "source": func_source,
                        "args": [a.arg for a in child.args.args],
                        "line_start": line_start,
                        "line_end": line_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "content_hash": self._hash_source(func_source) if func_source else None,
                    }
                )
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.Import):
                    names = [n.name for n in child.names]
                else:
                    names = [f"{child.module}.{n.name}" for n in child.names]
                summary["imports"].extend(names)

        for n in ast.walk(node):
            source_owner = owner_qualified_name_for(n)
            if isinstance(n, ast.Import):
                for alias in n.names:
                    target = alias.name
                    summary["imports"].append(target)
                    add_edge("imports", source_owner, target)
            elif isinstance(n, ast.ImportFrom):
                module_name = n.module or ""
                for alias in n.names:
                    target = f"{module_name}.{alias.name}" if module_name else alias.name
                    summary["imports"].append(target)
                    add_edge("imports", source_owner, target)
            elif isinstance(n, ast.Call):
                called = expr_to_name(n.func)
                if called:
                    add_edge("calls", source_owner, called)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in builtin_names:
                    continue
                use_key = ("uses", source_owner, n.id)
                if use_key in use_set:
                    continue
                use_set.add(use_key)
                add_edge("uses", source_owner, n.id)

            if isinstance(n, ast.ClassDef):
                add_node(n, "class", n.name)
            elif isinstance(n, ast.FunctionDef):
                if isinstance(parent_map.get(n), ast.ClassDef):
                    add_node(n, "method", n.name)
                else:
                    add_node(n, "function", n.name)
            elif isinstance(n, ast.AsyncFunctionDef):
                add_node(n, "function", n.name)
            elif isinstance(n, ast.For):
                add_node(n, "for_block", "for")
            elif isinstance(n, ast.While):
                add_node(n, "while_block", "while")
            elif isinstance(n, ast.If):
                add_node(n, "if_block", "if")
            elif isinstance(n, ast.With):
                add_node(n, "with_block", "with")
            elif isinstance(n, ast.Try):
                add_node(n, "try_block", "try")
            elif isinstance(n, ast.Match):
                add_node(n, "match_block", "match")
            elif isinstance(n, ast.BoolOp):
                add_node(n, "bool_op", "bool_op")
            elif isinstance(n, ast.ListComp):
                add_node(n, "list_comp", "list_comp")
            elif isinstance(n, ast.DictComp):
                add_node(n, "dict_comp", "dict_comp")
            elif isinstance(n, ast.SetComp):
                add_node(n, "set_comp", "set_comp")
            elif isinstance(n, ast.GeneratorExp):
                add_node(n, "gen_exp", "gen_exp")
            elif isinstance(n, ast.Lambda):
                add_node(n, "lambda", "lambda")

        # Preserve import order while removing duplicates.
        seen_imports: set[str] = set()
        unique_imports: list[str] = []
        for imp in summary["imports"]:
            if imp in seen_imports:
                continue
            seen_imports.add(imp)
            unique_imports.append(imp)
        summary["imports"] = unique_imports

        return summary
