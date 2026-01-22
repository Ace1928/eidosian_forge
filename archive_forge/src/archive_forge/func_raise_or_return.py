import locale
import pytest
from pandas._config import detect_console_encoding
@staticmethod
def raise_or_return(val):
    if isinstance(val, str):
        return val
    else:
        raise val