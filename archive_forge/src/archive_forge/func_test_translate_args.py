from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os
from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
from sympy.external import import_module
def test_translate_args():
    try:
        translate(None, None, None, 'not_none')
    except ValueError:
        pass
    else:
        assert False
    assert translate('s', None, None, None) == 's'
    try:
        translate('s', 'a', 'bc')
    except ValueError:
        pass
    else:
        assert False