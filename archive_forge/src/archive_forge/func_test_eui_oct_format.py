import pickle
import random
from netaddr import (
def test_eui_oct_format():
    assert oct(EUI('00-1B-77-49-54-FD')) == '0o1556722252375'