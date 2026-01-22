import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_trivial_function(self):

    def do_not_call(x):
        raise ArgmapError('do not call this function')

    @argmap(do_not_call)
    def trivial_argmap():
        return 1
    assert trivial_argmap() == 1