from typing import Iterator, Tuple, Union
from ...errors import Errors
from ...symbols import AUX, NOUN, PRON, PROPN, VERB
from ...tokens import Doc, Span
def get_bounds(doc, root):
    return (get_left_bound(root), get_right_bound(doc, root))