import collections
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def populate_db(parse_db):
    parse_db['set'] = parse_set