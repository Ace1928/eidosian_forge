import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_writer_kwarg_fobj(self):
    self.writer_kwarg(path=self.fobj)
    self.fobj.close()
    assert self.read(self.name) == ''.join(self.text)