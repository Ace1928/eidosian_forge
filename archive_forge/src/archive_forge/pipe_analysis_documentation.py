from typing import TYPE_CHECKING, Dict, ItemsView, Iterable, List, Set, Union
from wasabi import msg
from .errors import Errors
from .tokens import Doc, Span, Token
from .util import dot_to_dict
Print a formatted version of the pipe analysis produced by analyze_pipes.

    analysis (Dict[str, Union[List[str], Dict[str, List[str]]]]): The analysis.
    keys (List[str]): The meta keys to show in the table.
    