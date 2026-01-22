import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def next_ident_or_star(self) -> Optional[str]:
    next = self.next()
    if next.type == 'IDENT':
        return next.value
    elif next == ('DELIM', '*'):
        return None
    else:
        raise SelectorSyntaxError("Expected ident or '*', got %s" % (next,))