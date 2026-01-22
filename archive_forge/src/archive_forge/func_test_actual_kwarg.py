import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_actual_kwarg(self):

    @argmap(lambda x: -x, 'arg')
    def foo(*, arg):
        return arg
    assert foo(arg=3) == -3