import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
def test_make_repository(self):
    out, err = self.run_bzr('init-shared-repository --format=weave a')
    self.assertEqual(out, 'Standalone tree (format: weave)\nLocation:\n  branch root: a\n')
    self.assertEqual(err, '')