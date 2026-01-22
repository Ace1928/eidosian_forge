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
def visit_with(self, node, first_token, last_token):
    first = self._code.find_token(first_token, token.NAME, 'with', reverse=True)
    return (first, last_token)