from logging import error
import os
import sys
from IPython.core.error import TryNext, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import SList, strip_email_quotes
from IPython.utils import py3compat
@skip_doctest
@line_magic
def cpaste(self, parameter_s=''):
    """Paste & execute a pre-formatted code block from clipboard.

        You must terminate the block with '--' (two minus-signs) or Ctrl-D
        alone on the line. You can also provide your own sentinel with '%paste
        -s %%' ('%%' is the new sentinel for this operation).

        The block is dedented prior to execution to enable execution of method
        definitions. '>' and '+' characters at the beginning of a line are
        ignored, to allow pasting directly from e-mails, diff files and
        doctests (the '...' continuation prompt is also stripped).  The
        executed block is also assigned to variable named 'pasted_block' for
        later editing with '%edit pasted_block'.

        You can also pass a variable name as an argument, e.g. '%cpaste foo'.
        This assigns the pasted block to variable 'foo' as string, without
        dedenting or executing it (preceding >>> and + is still stripped)

        '%cpaste -r' re-executes the block previously entered by cpaste.
        '%cpaste -q' suppresses any additional output messages.

        Do not be alarmed by garbled output on Windows (it's a readline bug).
        Just press enter and type -- (and press enter again) and the block
        will be what was just pasted.

        Shell escapes are not supported (yet).

        See Also
        --------
        paste : automatically pull code from clipboard.

        Examples
        --------
        ::

          In [8]: %cpaste
          Pasting code; enter '--' alone on the line to stop.
          :>>> a = ["world!", "Hello"]
          :>>> print(" ".join(sorted(a)))
          :--
          Hello world!

        ::
          In [8]: %cpaste
          Pasting code; enter '--' alone on the line to stop.
          :>>> %alias_magic t timeit
          :>>> %t -n1 pass
          :--
          Created `%t` as an alias for `%timeit`.
          Created `%%t` as an alias for `%%timeit`.
          354 ns ± 224 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)
        """
    opts, name = self.parse_options(parameter_s, 'rqs:', mode='string')
    if 'r' in opts:
        self.rerun_pasted()
        return
    quiet = 'q' in opts
    sentinel = opts.get('s', u'--')
    block = '\n'.join(get_pasted_lines(sentinel, quiet=quiet))
    self.store_or_execute(block, name, store_history=True)