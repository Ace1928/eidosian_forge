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
def test_translate_included_attribute_text(self):
    """
        Verify that translated attributes end up in a proper `Attrs` instance.
        """
    html = HTML(u'<html>\n          <span title="Foo"></span>\n        </html>')
    translator = Translator(lambda s: u'Voh')
    stream = list(html.filter(translator))
    kind, data, pos = stream[2]
    assert isinstance(data[1], Attrs)