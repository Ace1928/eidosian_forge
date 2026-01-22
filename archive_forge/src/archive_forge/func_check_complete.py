from __future__ import annotations
from warnings import warn
import ast
import codeop
import io
import re
import sys
import tokenize
import warnings
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from types import CodeType
from IPython.core.inputtransformer import (leading_indent,
from IPython.utils import tokenutil
from IPython.core.inputtransformer import (ESC_SHELL, ESC_SH_CAP, ESC_HELP,
def check_complete(self, source):
    """Return whether a block of code is ready to execute, or should be continued

        This is a non-stateful API, and will reset the state of this InputSplitter.

        Parameters
        ----------
        source : string
            Python input code, which can be multiline.

        Returns
        -------
        status : str
            One of 'complete', 'incomplete', or 'invalid' if source is not a
            prefix of valid code.
        indent_spaces : int or None
            The number of spaces by which to indent the next line of code. If
            status is not 'incomplete', this is None.
        """
    self.reset()
    try:
        self.push(source)
    except SyntaxError:
        return ('invalid', None)
    else:
        if self._is_invalid:
            return ('invalid', None)
        elif self.push_accepts_more():
            return ('incomplete', self.get_indent_spaces())
        else:
            return ('complete', None)
    finally:
        self.reset()