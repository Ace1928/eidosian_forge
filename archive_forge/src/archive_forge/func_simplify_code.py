import ast
from typing import Any, List, Tuple
from langchain_community.document_loaders.parsers.language.code_segmenter import (
def simplify_code(self) -> str:
    tree = ast.parse(self.code)
    simplified_lines = self.source_lines[:]
    indices_to_del: List[Tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = (node.lineno - 1, node.end_lineno)
            simplified_lines[start] = f'# Code for: {simplified_lines[start]}'
            assert isinstance(end, int)
            indices_to_del.append((start + 1, end))
    for start, end in reversed(indices_to_del):
        del simplified_lines[start + 0:end]
    return '\n'.join(simplified_lines)