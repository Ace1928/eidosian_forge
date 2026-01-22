from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def yaml_end_comment_extend(self, comment, clear=False):
    if comment is None:
        return
    if clear or self.ca.end is None:
        self.ca.end = []
    self.ca.end.extend(comment)