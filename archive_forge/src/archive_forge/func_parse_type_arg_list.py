from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_type_arg_list(self):
    """
        type_arg_list : type_arg COMMA type_arg_list
                      | type_kwarg_list
                      | type_arg
        type_kwarg_list : type_kwarg COMMA type_kwarg_list
                        | type_kwarg

        Returns a tuple (args, kwargs), or (None, None).
        """
    args = []
    arg = True
    while arg is not None:
        arg = self.parse_type_arg()
        if arg is not None:
            if self.tok.id == lexer.COMMA:
                self.advance_tok()
                args.append(arg)
            else:
                args.append(arg)
                return (args, {})
        else:
            break
    kwargs = self.parse_homogeneous_list(self.parse_type_kwarg, lexer.COMMA, 'Expected another keyword argument, ' + 'positional arguments cannot follow ' + 'keyword arguments')
    return (args, dict(kwargs) if kwargs else {})