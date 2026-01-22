import io
import sys, os
from types import GenericAlias
def openhook(filename, mode):
    return open(filename, mode, encoding=encoding, errors=errors)