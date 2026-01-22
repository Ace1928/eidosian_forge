import ast
import sys
import dis
from types import CodeType, FrameType
from typing import Any, Callable, Iterator, Optional, Sequence, Set, Tuple, Type, Union, cast
from .executing import EnhancedAST, NotOneValueFound, Source, only, function_node_types, assert_
from ._exceptions import KnownIssue, VerifierFailure
from functools import lru_cache
def inst_match(opnames: Union[str, Sequence[str]], **kwargs: Any) -> bool:
    """
            match instruction

            Parameters:
                opnames: (str|Seq[str]): inst.opname has to be equal to or in `opname`
                **kwargs: every arg has to match inst.arg

            Returns:
                True if all conditions match the instruction

            """
    if isinstance(opnames, str):
        opnames = [opnames]
    return instruction.opname in opnames and kwargs == {k: getattr(instruction, k) for k in kwargs}