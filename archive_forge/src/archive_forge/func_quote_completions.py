from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def quote_completions(self, completions, cword_prequote, last_wordbreak_pos):
    """
        If the word under the cursor started with a quote (as indicated by a nonempty ``cword_prequote``), escapes
        occurrences of that quote character in the completions, and adds the quote to the beginning of each completion.
        Otherwise, escapes all characters that bash splits words on (``COMP_WORDBREAKS``), and removes portions of
        completions before the first colon if (``COMP_WORDBREAKS``) contains a colon.

        If there is only one completion, and it doesn't end with a **continuation character** (``/``, ``:``, or ``=``),
        adds a space after the completion.

        This method is exposed for overriding in subclasses; there is no need to use it directly.
        """
    special_chars = '\\'
    if cword_prequote == '':
        if last_wordbreak_pos:
            completions = [c[last_wordbreak_pos + 1:] for c in completions]
        special_chars += '();<>|&!`$* \t\n"\''
    elif cword_prequote == '"':
        special_chars += '"`$!'
    if os.environ.get('_ARGCOMPLETE_SHELL') == 'tcsh':
        special_chars = ''
    elif cword_prequote == "'":
        special_chars = ''
        completions = [c.replace("'", "'\\''") for c in completions]
    for char in special_chars:
        completions = [c.replace(char, '\\' + char) for c in completions]
    if self.append_space:
        continuation_chars = '=/:'
        if len(completions) == 1 and completions[0][-1] not in continuation_chars:
            if cword_prequote == '':
                completions[0] += ' '
    return completions