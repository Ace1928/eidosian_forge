import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
@functools.wraps(func)
def cmd_wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Optional[bool]:
    """
            Command function wrapper which translates command line into argparse Namespace and calls actual
            command function

            :param args: All positional arguments to this function.  We're expecting there to be:
                            cmd2_app, statement: Union[Statement, str]
                            contiguously somewhere in the list
            :param kwargs: any keyword arguments being passed to command function
            :return: return value of command function
            :raises: Cmd2ArgparseError if argparse has error parsing command line
            """
    cmd2_app, statement_arg = _parse_positionals(args)
    statement, parsed_arglist = cmd2_app.statement_parser.get_command_arg_list(command_name, statement_arg, preserve_quotes)
    if ns_provider is None:
        namespace = None
    else:
        provider_self = cmd2_app._resolve_func_self(ns_provider, args[0])
        namespace = ns_provider(provider_self if provider_self is not None else cmd2_app)
    try:
        new_args: Union[Tuple[argparse.Namespace], Tuple[argparse.Namespace, List[str]]]
        if with_unknown_args:
            new_args = parser.parse_known_args(parsed_arglist, namespace)
        else:
            new_args = (parser.parse_args(parsed_arglist, namespace),)
        ns = new_args[0]
    except SystemExit:
        raise Cmd2ArgparseError
    else:
        setattr(ns, 'cmd2_statement', Cmd2AttributeWrapper(statement))
        handler = getattr(ns, constants.NS_ATTR_SUBCMD_HANDLER, None)
        setattr(ns, 'cmd2_handler', Cmd2AttributeWrapper(handler))
        if hasattr(ns, constants.NS_ATTR_SUBCMD_HANDLER):
            delattr(ns, constants.NS_ATTR_SUBCMD_HANDLER)
        args_list = _arg_swap(args, statement_arg, *new_args)
        return func(*args_list, **kwargs)