import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def make_commit(self, **kwargs):
    attrs = {'tree': b'd80c186a03f423a81b39df39dc87fd269736ca86', 'parents': [b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'], 'author': b'James Westby <jw+debian@jameswestby.net>', 'committer': b'James Westby <jw+debian@jameswestby.net>', 'commit_time': 1174773719, 'author_time': 1174773719, 'commit_timezone': 0, 'author_timezone': 0, 'message': b'Merge ../b\n'}
    attrs.update(kwargs)
    return make_commit(**attrs)