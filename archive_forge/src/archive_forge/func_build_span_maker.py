from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
from thinc.api import (
from thinc.types import Floats2d
from ...errors import Errors
from ...kb import (
from ...tokens import Doc, Span
from ...util import registry
from ...vocab import Vocab
from ..extract_spans import extract_spans
def build_span_maker(n_sents: int=0) -> Model:
    model: Model = Model('span_maker', forward=span_maker_forward)
    model.attrs['n_sents'] = n_sents
    return model