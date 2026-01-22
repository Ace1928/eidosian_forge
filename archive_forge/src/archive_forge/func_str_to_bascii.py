import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def str_to_bascii(s):
    assert isinstance(s, str)
    return s