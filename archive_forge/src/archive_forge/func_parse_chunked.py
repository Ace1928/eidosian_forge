import io
import sys
from gunicorn.http.errors import (NoMoreData, ChunkMissingTerminator,
def parse_chunked(self, unreader):
    size, rest = self.parse_chunk_size(unreader)
    while size > 0:
        while size > len(rest):
            size -= len(rest)
            yield rest
            rest = unreader.read()
            if not rest:
                raise NoMoreData()
        yield rest[:size]
        rest = rest[size:]
        while len(rest) < 2:
            rest += unreader.read()
        if rest[:2] != b'\r\n':
            raise ChunkMissingTerminator(rest[:2])
        size, rest = self.parse_chunk_size(unreader, data=rest[2:])