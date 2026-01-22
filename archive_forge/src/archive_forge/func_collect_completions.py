from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def collect_completions(self, active_parsers, parsed_args, cword_prefix, debug):
    """
        Visits the active parsers and their actions, executes their completers or introspects them to collect their
        option strings. Returns the resulting completions as a list of strings.

        This method is exposed for overriding in subclasses; there is no need to use it directly.
        """
    completions = []
    debug('all active parsers:', active_parsers)
    active_parser = active_parsers[-1]
    debug('active_parser:', active_parser)
    if self.always_complete_options or (len(cword_prefix) > 0 and cword_prefix[0] in active_parser.prefix_chars):
        completions += self._get_option_completions(active_parser, cword_prefix)
    debug('optional options:', completions)
    next_positional = self._get_next_positional()
    debug('next_positional:', next_positional)
    if isinstance(next_positional, argparse._SubParsersAction):
        completions += self._get_subparser_completions(next_positional, cword_prefix)
    completions = self._complete_active_option(active_parser, next_positional, cword_prefix, parsed_args, completions)
    debug('active options:', completions)
    debug('display completions:', self._display_completions)
    return completions