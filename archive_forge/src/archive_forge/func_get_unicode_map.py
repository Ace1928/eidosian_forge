import gzip
import logging
import os
import os.path
import pickle as pickle
import struct
import sys
from typing import (
from .encodingdb import name2unicode
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSLiteral
from .psparser import PSStackParser
from .psparser import PSSyntaxError
from .psparser import literal_name
from .utils import choplist
from .utils import nunpack
@classmethod
def get_unicode_map(cls, name: str, vertical: bool=False) -> UnicodeMap:
    try:
        return cls._umap_cache[name][vertical]
    except KeyError:
        pass
    data = cls._load_data('to-unicode-%s' % name)
    cls._umap_cache[name] = [PyUnicodeMap(name, data, v) for v in (False, True)]
    return cls._umap_cache[name][vertical]