import unittest
from os.path import abspath, dirname, join
import errno
import os
def test_dict_storage(self):
    from kivy.storage.dictstore import DictStore
    from tempfile import mkstemp
    from os import unlink, close
    try:
        tmpfd, tmpfn = mkstemp('.dict')
        close(tmpfd)
        self._do_store_test_empty(DictStore(tmpfn))
        self._do_store_test_filled(DictStore(tmpfn))
    finally:
        unlink(tmpfn)