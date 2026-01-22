from __future__ import annotations
from collections.abc import Sequence
from string import ascii_letters, digits, hexdigits
from urllib.parse import quote as encode_uri_component
def get_encode_cache(exclude: str) -> Sequence[str]:
    if exclude in encode_cache:
        return encode_cache[exclude]
    cache: list[str] = []
    encode_cache[exclude] = cache
    for i in range(128):
        ch = chr(i)
        if ch in ASCII_LETTERS_AND_DIGITS:
            cache.append(ch)
        else:
            cache.append('%' + ('0' + hex(i)[2:].upper())[-2:])
    for i in range(len(exclude)):
        cache[ord(exclude[i])] = exclude[i]
    return cache