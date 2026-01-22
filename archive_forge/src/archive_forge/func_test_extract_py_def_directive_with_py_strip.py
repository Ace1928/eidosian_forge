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
def test_extract_py_def_directive_with_py_strip(self):
    tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/" py:strip="">\n    <py:def function="diff_options_fields(diff)">\n    <label for="style">View differences</label>\n    <select id="style" name="style">\n      <option selected="${diff.style == \'inline\' or None}"\n              value="inline">inline</option>\n      <option selected="${diff.style == \'sidebyside\' or None}"\n              value="sidebyside">side by side</option>\n    </select>\n    <div class="field">\n      Show <input type="text" name="contextlines" id="contextlines" size="2"\n                  maxlength="3" value="${diff.options.contextlines &lt; 0 and \'all\' or diff.options.contextlines}" />\n      <label for="contextlines">lines around each change</label>\n    </div>\n    <fieldset id="ignore" py:with="options = diff.options">\n      <legend>Ignore:</legend>\n      <div class="field">\n        <input type="checkbox" id="ignoreblanklines" name="ignoreblanklines"\n               checked="${options.ignoreblanklines or None}" />\n        <label for="ignoreblanklines">Blank lines</label>\n      </div>\n      <div class="field">\n        <input type="checkbox" id="ignorecase" name="ignorecase"\n               checked="${options.ignorecase or None}" />\n        <label for="ignorecase">Case changes</label>\n      </div>\n      <div class="field">\n        <input type="checkbox" id="ignorewhitespace" name="ignorewhitespace"\n               checked="${options.ignorewhitespace or None}" />\n        <label for="ignorewhitespace">White space changes</label>\n      </div>\n    </fieldset>\n    <div class="buttons">\n      <input type="submit" name="update" value="${_(\'Update\')}" />\n    </div>\n  </py:def></html>')
    translator = Translator()
    tmpl.add_directives(Translator.NAMESPACE, translator)
    messages = list(translator.extract(tmpl.stream))
    self.assertEqual(10, len(messages))
    self.assertEqual([(3, None, 'View differences', []), (6, None, 'inline', []), (8, None, 'side by side', []), (10, None, 'Show', []), (13, None, 'lines around each change', []), (16, None, 'Ignore:', []), (20, None, 'Blank lines', []), (25, None, 'Case changes', []), (30, None, 'White space changes', []), (34, '_', 'Update', [])], messages)