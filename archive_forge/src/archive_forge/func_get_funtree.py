import importlib
import logging
from cmakelang import lex
from cmakelang.parse.additional_nodes import ShellCommandNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, KwargBreaker, TreeNode
from cmakelang.parse.util import (
from cmakelang.parse.funs import standard_funs
def get_funtree(cmdspec):
    kwargs = {}
    for kwarg, subspec in cmdspec.kwargs.items():
        if kwarg in ('if', 'elseif', 'while'):
            subparser = ConditionalGroupNode.parse
        elif kwarg == 'COMMAND':
            subparser = ShellCommandNode.parse
        elif isinstance(subspec, standard_funs.CommandSpec):
            subparser = StandardParser2(subspec, get_funtree(subspec))
        else:
            raise ValueError('Unexpected kwarg spec of type {}'.format(type(subspec)))
        kwargs[kwarg] = subparser
    return kwargs