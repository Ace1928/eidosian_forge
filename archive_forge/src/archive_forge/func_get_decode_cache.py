from __future__ import annotations
from collections.abc import Sequence
import functools
import re
def get_decode_cache(exclude: str) -> Sequence[str]:
    if exclude in decode_cache:
        return decode_cache[exclude]
    cache: list[str] = []
    decode_cache[exclude] = cache
    for i in range(128):
        ch = chr(i)
        cache.append(ch)
    for i in range(len(exclude)):
        ch_code = ord(exclude[i])
        cache[ch_code] = '%' + ('0' + hex(ch_code)[2:].upper())[-2:]
    return cache