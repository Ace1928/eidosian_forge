import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def test_parents(self):
    self.assertSerialize(b' '.join([makesha(0), makesha(1), makesha(2)]), {makesha(0): [makesha(1), makesha(2)]})