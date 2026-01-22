import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_not_implemented_decorator_raise():
    with pytest.raises(nx.NetworkXNotImplemented):

        @not_implemented_for('graph')
        def test1(G):
            pass
        test1(nx.Graph())