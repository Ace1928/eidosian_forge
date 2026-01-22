import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def test_annotate_retries(self):
    vf, reload_counter = self.make_vf_for_retrying()
    key = (b'rev-3',)
    reload_lines = vf.annotate(key)
    self.assertEqual([1, 1, 0], reload_counter)
    plain_lines = vf.annotate(key)
    self.assertEqual([1, 1, 0], reload_counter)
    if reload_lines != plain_lines:
        self.fail('Annotation was not identical with reloading.')
    for trans, name in vf._access._indices.values():
        trans.delete(name)
    self.assertRaises(_mod_transport.NoSuchFile, vf.annotate, key)
    self.assertEqual([2, 1, 1], reload_counter)