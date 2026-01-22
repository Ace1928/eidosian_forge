import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def uascii_to_str(s):
    assert isinstance(s, unicode)
    return s.encode('ascii')