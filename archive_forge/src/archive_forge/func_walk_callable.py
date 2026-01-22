import ast
import hashlib
import inspect
import os
import re
import warnings
from collections import deque
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import (
from w3lib.html import replace_entities
from scrapy.item import Item
from scrapy.utils.datatypes import LocalWeakReferencedCache
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.python import flatten, to_unicode
def walk_callable(node: ast.AST) -> Generator[ast.AST, Any, None]:
    """Similar to ``ast.walk``, but walks only function body and skips nested
    functions defined within the node.
    """
    todo: Deque[ast.AST] = deque([node])
    walked_func_def = False
    while todo:
        node = todo.popleft()
        if isinstance(node, ast.FunctionDef):
            if walked_func_def:
                continue
            walked_func_def = True
        todo.extend(ast.iter_child_nodes(node))
        yield node