from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
def video_filter():
    frame = (yield None)
    while frame is not None:
        graph.push(frame)
        try:
            frame = (yield graph.pull())
        except av.error.BlockingIOError:
            frame = (yield None)
        except av.error.EOFError:
            break
    try:
        graph.push(None)
    except ValueError:
        pass
    while True:
        try:
            yield graph.pull()
        except av.error.EOFError:
            break
        except av.error.BlockingIOError:
            break