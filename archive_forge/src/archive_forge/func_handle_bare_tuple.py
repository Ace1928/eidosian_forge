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
def handle_bare_tuple(self, node, first_token, last_token):
    maybe_comma = self._code.next_token(last_token)
    if util.match_token(maybe_comma, token.OP, ','):
        last_token = maybe_comma
    return (first_token, last_token)