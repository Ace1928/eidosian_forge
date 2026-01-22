from __future__ import annotations
from collections.abc import Sequence
import functools
import re
def repl_func_with_cache(match: re.Match, cache: Sequence[str]) -> str:
    seq = match.group()
    result = ''
    i = 0
    l = len(seq)
    while i < l:
        b1 = int(seq[i + 1:i + 3], 16)
        if b1 < 128:
            result += cache[b1]
            i += 3
            continue
        if b1 & 224 == 192 and i + 3 < l:
            b2 = int(seq[i + 4:i + 6], 16)
            if b2 & 192 == 128:
                all_bytes = bytes((b1, b2))
                try:
                    result += all_bytes.decode()
                except UnicodeDecodeError:
                    result += '�' * 2
                i += 3
                i += 3
                continue
        if b1 & 240 == 224 and i + 6 < l:
            b2 = int(seq[i + 4:i + 6], 16)
            b3 = int(seq[i + 7:i + 9], 16)
            if b2 & 192 == 128 and b3 & 192 == 128:
                all_bytes = bytes((b1, b2, b3))
                try:
                    result += all_bytes.decode()
                except UnicodeDecodeError:
                    result += '�' * 3
                i += 6
                i += 3
                continue
        if b1 & 248 == 240 and i + 9 < l:
            b2 = int(seq[i + 4:i + 6], 16)
            b3 = int(seq[i + 7:i + 9], 16)
            b4 = int(seq[i + 10:i + 12], 16)
            if b2 & 192 == 128 and b3 & 192 == 128 and (b4 & 192 == 128):
                all_bytes = bytes((b1, b2, b3, b4))
                try:
                    result += all_bytes.decode()
                except UnicodeDecodeError:
                    result += '�' * 4
                i += 9
                i += 3
                continue
        result += '�'
        i += 3
    return result