import re
from .. import osutils
from ..iterablefile import IterableFile
def str_iter():
    if header is not None:
        yield (header + b'\n')
    first_stanza = True
    for s in stanzas:
        if first_stanza is not True:
            yield b'\n'
        yield from s.to_lines()
        first_stanza = False