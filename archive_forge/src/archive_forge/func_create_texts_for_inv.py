from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def create_texts_for_inv(repo, inv):
    for path, ie in inv.iter_entries():
        if ie.text_size:
            lines = [b'a' * ie.text_size]
        else:
            lines = []
        repo.texts.add_lines((ie.file_id, ie.revision), [], lines)