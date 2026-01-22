import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def with_argument_list(func_arg: Optional[ArgListCommandFunc]=None, *, preserve_quotes: bool=False) -> Union[RawCommandFuncOptionalBoolReturn, Callable[[ArgListCommandFunc], RawCommandFuncOptionalBoolReturn]]:
    """
    A decorator to alter the arguments passed to a ``do_*`` method. Default
    passes a string of whatever the user typed. With this decorator, the
    decorated method will receive a list of arguments parsed from user input.

    :param func_arg: Single-element positional argument list containing ``do_*`` method
                 this decorator is wrapping
    :param preserve_quotes: if ``True``, then argument quotes will not be stripped
    :return: function that gets passed a list of argument strings

    :Example:

    >>> class MyApp(cmd2.Cmd):
    >>>     @cmd2.with_argument_list
    >>>     def do_echo(self, arglist):
    >>>         self.poutput(' '.join(arglist)
    """
    import functools

    def arg_decorator(func: ArgListCommandFunc) -> RawCommandFuncOptionalBoolReturn:
        """
        Decorator function that ingests an Argument List function and returns a raw command function.
        The returned function will process the raw input into an argument list to be passed to the wrapped function.

        :param func: The defined argument list command function
        :return: Function that takes raw input and converts to an argument list to pass to the wrapped function.
        """

        @functools.wraps(func)
        def cmd_wrapper(*args: Any, **kwargs: Any) -> Optional[bool]:
            """
            Command function wrapper which translates command line into an argument list and calls actual command function

            :param args: All positional arguments to this function.  We're expecting there to be:
                            cmd2_app, statement: Union[Statement, str]
                            contiguously somewhere in the list
            :param kwargs: any keyword arguments being passed to command function
            :return: return value of command function
            """
            cmd2_app, statement = _parse_positionals(args)
            _, parsed_arglist = cmd2_app.statement_parser.get_command_arg_list(command_name, statement, preserve_quotes)
            args_list = _arg_swap(args, statement, parsed_arglist)
            return func(*args_list, **kwargs)
        command_name = func.__name__[len(constants.COMMAND_FUNC_PREFIX):]
        cmd_wrapper.__doc__ = func.__doc__
        return cmd_wrapper
    if callable(func_arg):
        return arg_decorator(func_arg)
    else:
        return arg_decorator