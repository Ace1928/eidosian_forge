import glob
import json
import os
import unittest
import pytest
from monty.serialization import dumpfn, loadfn
from monty.tempfile import ScratchDir
def test_dumpfn_loadfn(self):
    d = {'hello': 'world'}
    for ext in ('json', 'yaml', 'yml', 'json.gz', 'yaml.gz', 'json.bz2', 'yaml.bz2'):
        fn = f'monte_test.{ext}'
        dumpfn(d, fn)
        d2 = loadfn(fn)
        assert d == d2, f'Test file with extension {ext} did not parse correctly'
        os.remove(fn)
    dumpfn(d, 'monte_test.json', indent=4)
    d2 = loadfn('monte_test.json')
    assert d == d2
    os.remove('monte_test.json')
    dumpfn(d, 'monte_test.yaml')
    d2 = loadfn('monte_test.yaml')
    assert d == d2
    os.remove('monte_test.yaml')
    dumpfn(d, 'monte_test.json', fmt='yaml')
    with pytest.raises(json.decoder.JSONDecodeError):
        loadfn('monte_test.json')
    d2 = loadfn('monte_test.json', fmt='yaml')
    assert d == d2
    os.remove('monte_test.json')
    with pytest.raises(TypeError):
        dumpfn(d, 'monte_test.txt', fmt='garbage')
    with pytest.raises(TypeError):
        loadfn('monte_test.txt', fmt='garbage')