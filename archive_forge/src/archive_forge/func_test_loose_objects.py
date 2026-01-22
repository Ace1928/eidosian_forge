import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
def test_loose_objects(self):
    expected_shas = self._get_loose_shas()
    self.assertShasMatch(expected_shas, self._repo.object_store._iter_loose_objects())