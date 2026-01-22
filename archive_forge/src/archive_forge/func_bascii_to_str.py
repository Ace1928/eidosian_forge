import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def bascii_to_str(s):
    assert isinstance(s, bytes)
    return s