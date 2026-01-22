import matplotlib._type1font as t1f
import os.path
import difflib
import pytest
def bin_after(n):
    tokens = t1f._tokenize(data, True)
    result = []
    for _ in range(n):
        result.append(next(tokens))
    result.append(tokens.send(10))
    return convert(result)