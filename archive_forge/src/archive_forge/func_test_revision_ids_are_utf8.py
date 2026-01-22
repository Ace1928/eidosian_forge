from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_revision_ids_are_utf8(self):
    """Parsed revision_ids should all be utf-8 strings, not unicode."""
    s_v5 = breezy.bzr.xml5.serializer_v5
    rev = s_v5.read_revision_from_string(_revision_utf8_v5)
    self.assertEqual(b'erik@b\xc3\xa5gfors-02', rev.revision_id)
    self.assertIsInstance(rev.revision_id, bytes)
    self.assertEqual([b'erik@b\xc3\xa5gfors-01'], rev.parent_ids)
    for parent_id in rev.parent_ids:
        self.assertIsInstance(parent_id, bytes)
    self.assertEqual('Include µnicode characters\n', rev.message)
    self.assertIsInstance(rev.message, str)
    inv = s_v5.read_inventory_from_lines(breezy.osutils.split_lines(_inventory_utf8_v5))
    rev_id_1 = 'erik@bågfors-01'.encode()
    rev_id_2 = 'erik@bågfors-02'.encode()
    fid_root = 'TREé_ROOT'.encode()
    fid_bar1 = 'bår-01'.encode()
    fid_sub = 'sµbdir-01'.encode()
    fid_bar2 = 'bår-02'.encode()
    expected = [('', fid_root, None, rev_id_2), ('bår', fid_bar1, fid_root, rev_id_1), ('sµbdir', fid_sub, fid_root, rev_id_1), ('sµbdir/bår', fid_bar2, fid_sub, rev_id_2)]
    self.assertEqual(rev_id_2, inv.revision_id)
    self.assertIsInstance(inv.revision_id, bytes)
    actual = list(inv.iter_entries_by_dir())
    for (exp_path, exp_file_id, exp_parent_id, exp_rev_id), (act_path, act_ie) in zip(expected, actual):
        self.assertEqual(exp_path, act_path)
        self.assertIsInstance(act_path, str)
        self.assertEqual(exp_file_id, act_ie.file_id)
        self.assertIsInstance(act_ie.file_id, bytes)
        self.assertEqual(exp_parent_id, act_ie.parent_id)
        if exp_parent_id is not None:
            self.assertIsInstance(act_ie.parent_id, bytes)
        self.assertEqual(exp_rev_id, act_ie.revision)
        if exp_rev_id is not None:
            self.assertIsInstance(act_ie.revision, bytes)
    self.assertEqual(len(expected), len(actual))