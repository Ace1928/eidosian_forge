from datetime import datetime
import io
import numpy as np
from ase.io.jsonio import encode, decode, read_json, write_json
def test_dict_with_int_key():
    assert decode(encode({1: 2}), False)[1] == 2