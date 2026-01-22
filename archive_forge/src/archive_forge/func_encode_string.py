from typing import Dict, Type, Callable, List
def encode_string(x, r):
    r.extend((int_to_bytes(len(x)), b':', x))