import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
def received_frames(self) -> Generator[Frame, None, None]:
    for event in self._parse_more:
        if event is None:
            break
        else:
            yield event