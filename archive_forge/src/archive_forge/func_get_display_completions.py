from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def get_display_completions(self):
    """
        This function returns a mapping of option names to their help strings for displaying to the user

        Usage:

        .. code-block:: python

            def display_completions(substitution, matches, longest_match_length):
                _display_completions = argcomplete.autocomplete.get_display_completions()
                print("")
                if _display_completions:
                    help_len = [len(x) for x in _display_completions.values() if x]

                    if help_len:
                        maxlen = max([len(x) for x in _display_completions])
                        print("\\n".join("{0:{2}} -- {1}".format(k, v, maxlen)
                                        for k, v in sorted(_display_completions.items())))
                    else:
                        print("    ".join(k for k in sorted(_display_completions)))
                else:
                    print(" ".join(x for x in sorted(matches)))

                import readline
                print("cli /> {0}".format(readline.get_line_buffer()), end="")
                readline.redisplay()

            ...
            readline.set_completion_display_matches_hook(display_completions)

        """
    return self._display_completions