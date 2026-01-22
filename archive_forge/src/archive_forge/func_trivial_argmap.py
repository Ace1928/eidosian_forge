import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@argmap(do_not_call)
def trivial_argmap():
    yield from (1, 2, 3)