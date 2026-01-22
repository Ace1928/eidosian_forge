from __future__ import absolute_import, print_function, division
import io
import json
import inspect
from json.encoder import JSONEncoder
from os import unlink
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.compat import pickle
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.util.base import data, Table, dicts as _dicts, iterpeek
def iterjlines(f, header, missing):
    it = iter(f)
    if header is None:
        header = list()
        peek, it = iterpeek(it, 1)
        json_obj = json.loads(peek)
        if hasattr(json_obj, 'keys'):
            header += [k for k in json_obj.keys() if k not in header]
    yield tuple(header)
    for o in it:
        json_obj = json.loads(o)
        yield tuple((json_obj[f] if f in json_obj else missing for f in header))