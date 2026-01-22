import re
from .. import osutils
from ..iterablefile import IterableFile
def rio_file(stanzas, header=None):
    """Produce a rio IterableFile from an iterable of stanzas"""

    def str_iter():
        if header is not None:
            yield (header + b'\n')
        first_stanza = True
        for s in stanzas:
            if first_stanza is not True:
                yield b'\n'
            yield from s.to_lines()
            first_stanza = False
    return IterableFile(str_iter())