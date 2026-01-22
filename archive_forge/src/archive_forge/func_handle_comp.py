import ast
import numbers
import sys
import token
from ast import Module
from typing import Callable, List, Union, cast, Optional, Tuple, TYPE_CHECKING
import six
from . import util
from .asttokens import ASTTokens
from .util import AstConstant
from .astroid_compat import astroid_node_classes as nc, BaseContainer as AstroidBaseContainer
def handle_comp(self, open_brace, node, first_token, last_token):
    before = self._code.prev_token(first_token)
    util.expect_token(before, token.OP, open_brace)
    return (before, last_token)