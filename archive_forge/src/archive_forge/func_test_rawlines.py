from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os
from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
from sympy.external import import_module
def test_rawlines():
    assert rawlines('a a\na') == "dedent('''\\\n    a a\n    a''')"
    assert rawlines('a a') == "'a a'"
    assert rawlines(strlines('\\le"ft')) == '(\n    \'(\\n\'\n    \'r\\\'\\\\le"ft\\\'\\n\'\n    \')\'\n)'