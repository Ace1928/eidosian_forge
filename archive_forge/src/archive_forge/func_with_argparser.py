import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def with_argparser(parser: argparse.ArgumentParser, *, ns_provider: Optional[Callable[..., argparse.Namespace]]=None, preserve_quotes: bool=False, with_unknown_args: bool=False) -> Callable[[ArgparseCommandFunc], RawCommandFuncOptionalBoolReturn]:
    """A decorator to alter a cmd2 method to populate its ``args`` argument by parsing arguments
    with the given instance of argparse.ArgumentParser.

    :param parser: unique instance of ArgumentParser
    :param ns_provider: An optional function that accepts a cmd2.Cmd or cmd2.CommandSet object as an argument and returns an
                        argparse.Namespace. This is useful if the Namespace needs to be prepopulated with state data that
                        affects parsing.
    :param preserve_quotes: if ``True``, then arguments passed to argparse maintain their quotes
    :param with_unknown_args: if true, then capture unknown args
    :return: function that gets passed argparse-parsed args in a ``Namespace``
             A :class:`cmd2.argparse_custom.Cmd2AttributeWrapper` called ``cmd2_statement`` is included
             in the ``Namespace`` to provide access to the :class:`cmd2.Statement` object that was created when
             parsing the command line. This can be useful if the command function needs to know the command line.

    :Example:

    >>> parser = cmd2.Cmd2ArgumentParser()
    >>> parser.add_argument('-p', '--piglatin', action='store_true', help='atinLay')
    >>> parser.add_argument('-s', '--shout', action='store_true', help='N00B EMULATION MODE')
    >>> parser.add_argument('-r', '--repeat', type=int, help='output [n] times')
    >>> parser.add_argument('words', nargs='+', help='words to print')
    >>>
    >>> class MyApp(cmd2.Cmd):
    >>>     @cmd2.with_argparser(parser, preserve_quotes=True)
    >>>     def do_argprint(self, args):
    >>>         "Print the options and argument list this options command was called with."
    >>>         self.poutput(f'args: {args!r}')

    :Example with unknown args:

    >>> parser = cmd2.Cmd2ArgumentParser()
    >>> parser.add_argument('-p', '--piglatin', action='store_true', help='atinLay')
    >>> parser.add_argument('-s', '--shout', action='store_true', help='N00B EMULATION MODE')
    >>> parser.add_argument('-r', '--repeat', type=int, help='output [n] times')
    >>>
    >>> class MyApp(cmd2.Cmd):
    >>>     @cmd2.with_argparser(parser, with_unknown_args=True)
    >>>     def do_argprint(self, args, unknown):
    >>>         "Print the options and argument list this options command was called with."
    >>>         self.poutput(f'args: {args!r}')
    >>>         self.poutput(f'unknowns: {unknown}')

    """
    import functools

    def arg_decorator(func: ArgparseCommandFunc) -> RawCommandFuncOptionalBoolReturn:
        """
        Decorator function that ingests an Argparse Command Function and returns a raw command function.
        The returned function will process the raw input into an argparse Namespace to be passed to the wrapped function.

        :param func: The defined argparse command function
        :return: Function that takes raw input and converts to an argparse Namespace to passed to the wrapped function.
        """

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
        command_name = func.__name__[len(constants.COMMAND_FUNC_PREFIX):]
        _set_parser_prog(parser, command_name)
        if parser.description is None and func.__doc__:
            parser.description = strip_doc_annotations(func.__doc__)
        cmd_wrapper.__doc__ = parser.description
        setattr(cmd_wrapper, constants.CMD_ATTR_ARGPARSER, parser)
        setattr(cmd_wrapper, constants.CMD_ATTR_PRESERVE_QUOTES, preserve_quotes)
        return cmd_wrapper
    return arg_decorator