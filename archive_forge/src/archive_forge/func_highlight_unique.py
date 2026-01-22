import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def highlight_unique(lst: List[T]) -> Iterator[Tuple[T, bool]]:
    counts = Counter(lst)
    for is_common, group in itertools.groupby(lst, key=lambda x: counts[x] > 3):
        if is_common:
            group = list(group)
            highlighted = [False] * len(group)

            def highlight_index(f):
                try:
                    i = f()
                except ValueError:
                    return None
                highlighted[i] = True
                return i
            for item in set(group):
                first = highlight_index(lambda: group.index(item))
                if first is not None:
                    highlight_index(lambda: group.index(item, first + 1))
                highlight_index(lambda: -1 - group[::-1].index(item))
        else:
            highlighted = itertools.repeat(True)
        yield from zip(group, highlighted)