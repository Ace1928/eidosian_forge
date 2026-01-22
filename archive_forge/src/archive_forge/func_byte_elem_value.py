import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def byte_elem_value(elem):
    assert isinstance(elem, int)
    return elem