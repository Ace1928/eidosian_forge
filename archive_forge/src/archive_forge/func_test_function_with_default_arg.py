import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
def test_function_with_default_arg(self):
    """
        Verify that keyword arguments work with `py:def` directives.
        """
    tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <b py:def="echo(what, bold=False)" py:strip="not bold">${what}</b>\n          ${echo(\'foo\')}\n        </doc>')
    self.assertEqual('<doc>\n          foo\n        </doc>', tmpl.generate().render(encoding=None))