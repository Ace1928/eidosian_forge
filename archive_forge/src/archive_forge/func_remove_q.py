from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def remove_q(self) -> TextStateManagerChainMapType:
    """Rewind to stack prior state after closing a 'q' with internal 'cm' ops"""
    self.transform_stack = self.reset_tm()
    self.transform_stack.maps = self.transform_stack.maps[self.q_queue.pop(self.q_depth.pop(), 0):]
    return self.transform_stack