import shutil
from base64 import b64decode
from twisted.persisted import dirdbm
from twisted.python import rebuild
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_recovery(self) -> None:
    """
        DirDBM: test recovery from directory after a faked crash
        """
    k = self.dbm._encode(b'key1')
    with self.path.child(k + b'.rpl').open(mode='w') as f:
        f.write(b'value')
    k2 = self.dbm._encode(b'key2')
    with self.path.child(k2).open(mode='w') as f:
        f.write(b'correct')
    with self.path.child(k2 + b'.rpl').open(mode='w') as f:
        f.write(b'wrong')
    with self.path.child('aa.new').open(mode='w') as f:
        f.write(b'deleted')
    dbm = dirdbm.DirDBM(self.path.path)
    self.assertEqual(dbm[b'key1'], b'value')
    self.assertEqual(dbm[b'key2'], b'correct')
    self.assertFalse(self.path.globChildren('*.new'))
    self.assertFalse(self.path.globChildren('*.rpl'))