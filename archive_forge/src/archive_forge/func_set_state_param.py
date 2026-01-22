from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def set_state_param(self, op: bytes, value: Union[float, List[Any]]) -> None:
    """
        Set a text state parameter. Supports Tc, Tz, Tw, TL, and Ts operators.

        Args:
            op: operator read from PDF stream as bytes. No action is taken
                for unsupported operators (see supported operators above).
            value (float | List[Any]): new parameter value. If a list,
                value[0] is used.
        """
    if op not in [b'Tc', b'Tz', b'Tw', b'TL', b'Ts']:
        return
    self.__setattr__(op.decode(), value[0] if isinstance(value, list) else value)