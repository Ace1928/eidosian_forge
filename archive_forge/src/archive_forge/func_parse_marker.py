import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def parse_marker(source: str) -> MarkerList:
    return _parse_marker(Tokenizer(source, rules=DEFAULT_RULES))