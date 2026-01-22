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
def visit_starred(self, node, first_token, last_token):
    if not util.match_token(first_token, token.OP, '*'):
        star = self._code.prev_token(first_token)
        if util.match_token(star, token.OP, '*'):
            first_token = star
    return (first_token, last_token)