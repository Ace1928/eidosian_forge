import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_format_node():
    node = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='node')
    serialized = format_node(node)
    workspace = {'Node': pe.Node}
    exec('\n'.join(serialized), workspace)
    assert workspace['node'].interface._fields == node.interface._fields