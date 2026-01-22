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
def test_text_template_extraction(self):
    buf = StringIO('${_("Dear %(name)s") % {\'name\': name}},\n\n        ${ngettext("Your item:", "Your items", len(items))}\n        #for item in items\n         * $item\n        #end\n\n        All the best,\n        Foobar')
    results = list(extract(buf, ['_', 'ngettext'], [], {'template_class': 'genshi.template:TextTemplate'}))
    self.assertEqual([(1, '_', 'Dear %(name)s', []), (3, 'ngettext', ('Your item:', 'Your items', None), []), (7, None, 'All the best,\n        Foobar', [])], results)