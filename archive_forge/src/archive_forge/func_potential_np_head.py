from typing import Iterator, Tuple, Union
from ...errors import Errors
from ...symbols import NOUN, PRON, PROPN
from ...tokens import Doc, Span
def potential_np_head(word):
    return word.pos in (NOUN, PROPN) and (word.dep in np_deps or word.head.pos == PRON)