import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
def get_command_arg_list(self, command_name: str, to_parse: Union[Statement, str], preserve_quotes: bool) -> Tuple[Statement, List[str]]:
    """
        Convenience method used by the argument parsing decorators.

        Retrieves just the arguments being passed to their ``do_*`` methods as a list.

        :param command_name: name of the command being run
        :param to_parse: what is being passed to the ``do_*`` method. It can be one of two types:

                             1. An already parsed :class:`~cmd2.Statement`
                             2. An argument string in cases where a ``do_*`` method is
                                explicitly called. Calling ``do_help('alias create')`` would
                                cause ``to_parse`` to be 'alias create'.

                                In this case, the string will be converted to a
                                :class:`~cmd2.Statement` and returned along with
                                the argument list.

        :param preserve_quotes: if ``True``, then quotes will not be stripped from
                                the arguments
        :return: A tuple containing the :class:`~cmd2.Statement` and a list of
                 strings representing the arguments
        """
    if not isinstance(to_parse, Statement):
        to_parse = self.parse(command_name + ' ' + to_parse)
    if preserve_quotes:
        return (to_parse, to_parse.arg_list)
    else:
        return (to_parse, to_parse.argv[1:])