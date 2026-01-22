import glob
import json
import os
import unittest
import pytest
from monty.serialization import dumpfn, loadfn
from monty.tempfile import ScratchDir
@unittest.skipIf(msgpack is None, 'msgpack-python not installed.')
def test_mpk(self):
    d = {'hello': 'world'}
    dumpfn(d, 'monte_test.mpk')
    d2 = loadfn('monte_test.mpk')
    assert d, {k: v for k, v in d2.items()}
    os.remove('monte_test.mpk')
    with ScratchDir('.'):
        os.mkdir('mpk_test')
        os.chdir('mpk_test')
        fname = os.path.abspath('test_file.json')
        dumpfn({'test': 1}, fname)
        with open('test_file.json') as f:
            reloaded = json.loads(f.read())
        assert reloaded['test'] == 1