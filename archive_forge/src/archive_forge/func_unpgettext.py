from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
def unpgettext(self, context, msgid1, msgid2, n):
    try:
        return self._catalog[context, msgid1, self.plural(n)]
    except KeyError:
        if self._fallback:
            return self._fallback.unpgettext(context, msgid1, msgid2, n)
        if n == 1:
            return msgid1
        else:
            return msgid2