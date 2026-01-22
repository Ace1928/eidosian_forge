import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_flatten(self):
    assert tuple(argmap._flatten([[[[[], []], [], []], [], [], []]], set())) == ()
    rlist = ['a', ['b', 'c'], [['d'], 'e'], 'f']
    assert ''.join(argmap._flatten(rlist, set())) == 'abcdef'