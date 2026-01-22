import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@classmethod
def teardown_class(cls):
    os.unlink(cls.fname)