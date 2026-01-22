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
def test_deserialize_mergetags(self):
    tag = make_object(Tag, object=(Commit, b'a38d6181ff27824c79fc7df825164a212eff6a3f'), object_type_name=b'commit', name=b'v2.6.22-rc7', tag_time=1183319674, tag_timezone=0, tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org>', message=default_message)
    commit = self.make_commit(mergetag=[tag, tag])
    d = Commit()
    d._deserialize(commit.as_raw_chunks())
    self.assertEqual(commit, d)