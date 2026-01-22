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
def visit_asyncfunctiondef(self, node, first_token, last_token):
    if util.match_token(first_token, token.NAME, 'def'):
        first_token = self._code.prev_token(first_token)
    return self.visit_functiondef(node, first_token, last_token)