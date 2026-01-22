from __future__ import unicode_literals
from . import (lookup, LABELS, decode, encode, iter_decode, iter_encode,
def test_all_labels():
    for label in LABELS:
        assert decode(b'', label) == ('', lookup(label))
        assert encode('', label) == b''
        for repeat in [0, 1, 12]:
            output, _ = iter_decode([b''] * repeat, label)
            assert list(output) == []
            assert list(iter_encode([''] * repeat, label)) == []
        decoder = IncrementalDecoder(label)
        assert decoder.decode(b'') == ''
        assert decoder.decode(b'', final=True) == ''
        encoder = IncrementalEncoder(label)
        assert encoder.encode('') == b''
        assert encoder.encode('', final=True) == b''
    for name in set(LABELS.values()):
        assert lookup(name).name == name