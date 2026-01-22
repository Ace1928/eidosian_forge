from glob import glob
import os
from shutil import rmtree
from itertools import product
import pytest
import networkx as nx
from .... import config
from ....interfaces import utility as niu
from ... import engine as pe
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
def test_disconnect():
    a = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='a')
    b = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='b')
    flow1 = pe.Workflow(name='test')
    flow1.connect(a, 'a', b, 'a')
    flow1.disconnect(a, 'a', b, 'a')
    assert list(flow1._graph.edges()) == []