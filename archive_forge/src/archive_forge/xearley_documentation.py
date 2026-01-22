from typing import TYPE_CHECKING, Callable, Optional, List, Any
from collections import defaultdict
from ..tree import Tree
from ..exceptions import UnexpectedCharacters
from ..lexer import Token
from ..grammar import Terminal
from .earley import Parser as BaseParser
from .earley_forest import TokenNode
The core Earley Scanner.

            This is a custom implementation of the scanner that uses the
            Lark lexer to match tokens. The scan list is built by the
            Earley predictor, based on the previously completed tokens.
            This ensures that at each phase of the parse we have a custom
            lexer context, allowing for more complex ambiguities.