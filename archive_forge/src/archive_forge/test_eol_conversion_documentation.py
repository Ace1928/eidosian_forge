import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
Check the committed content and content in cloned trees.

        :param roundtrip_to: the set of formats (excluding exact) we
          can round-trip to or None for all
        