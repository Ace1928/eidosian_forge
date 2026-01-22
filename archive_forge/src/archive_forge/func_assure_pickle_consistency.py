import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def assure_pickle_consistency(verbose=False):
    copy = code2op.copy()
    for name in pickle.__all__:
        if not re.match('[A-Z][A-Z0-9_]+$', name):
            if verbose:
                print("skipping %r: it doesn't look like an opcode name" % name)
            continue
        picklecode = getattr(pickle, name)
        if not isinstance(picklecode, bytes) or len(picklecode) != 1:
            if verbose:
                print("skipping %r: value %r doesn't look like a pickle code" % (name, picklecode))
            continue
        picklecode = picklecode.decode('latin-1')
        if picklecode in copy:
            if verbose:
                print('checking name %r w/ code %r for consistency' % (name, picklecode))
            d = copy[picklecode]
            if d.name != name:
                raise ValueError("for pickle code %r, pickle.py uses name %r but we're using name %r" % (picklecode, name, d.name))
            del copy[picklecode]
        else:
            raise ValueError("pickle.py appears to have a pickle opcode with name %r and code %r, but we don't" % (name, picklecode))
    if copy:
        msg = ["we appear to have pickle opcodes that pickle.py doesn't have:"]
        for code, d in copy.items():
            msg.append('    name %r with code %r' % (d.name, code))
        raise ValueError('\n'.join(msg))