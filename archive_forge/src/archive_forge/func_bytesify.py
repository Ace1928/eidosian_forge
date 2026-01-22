from typing import Any, Dict, NoReturn, Pattern, Tuple, Type, TypeVar, Union
def bytesify(s: Union[bytes, bytearray, memoryview, int, str]) -> bytes:
    if type(s) is bytes:
        return s
    if isinstance(s, str):
        s = s.encode('ascii')
    if isinstance(s, int):
        raise TypeError('expected bytes-like object, not int')
    return bytes(s)