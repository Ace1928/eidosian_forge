import os, sys, time, random
def encodeBytes(self, src: bytes):
    idx: int = 0
    tokens = []
    while idx < len(src):
        _idx: int = idx
        idx, _, values = self.root.find_longest(src, idx)
        assert idx != _idx
        _, token = next(iter(values))
        tokens.append(token)
    return tokens