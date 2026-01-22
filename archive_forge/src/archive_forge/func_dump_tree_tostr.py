from __future__ import print_function
from __future__ import unicode_literals
import io
import sys
from cmakelang.parse.common import TreeNode
def dump_tree_tostr(nodes, indent=None):
    outfile = io.StringIO()
    dump_tree(nodes, outfile, indent)
    return outfile.getvalue()