import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def xform(x, y):
    u, v = y
    return (x + u + v, (x + u, x + v))