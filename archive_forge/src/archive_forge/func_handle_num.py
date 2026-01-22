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
def handle_num(self, node, value, first_token, last_token):
    while util.match_token(last_token, token.OP):
        last_token = self._code.next_token(last_token)
    if isinstance(value, complex):
        value = value.imag
    if value < 0 and first_token.type == token.NUMBER:
        first_token = self._code.prev_token(first_token)
    return (first_token, last_token)