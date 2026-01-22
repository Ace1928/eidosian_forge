import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_render_fragment_with_doctype(self):
    plugin = MarkupTemplateEnginePlugin(options={'genshi.default_doctype': 'html-strict'})
    tmpl = plugin.load_template(PACKAGE + '.templates.test_no_doctype')
    output = plugin.render({'message': 'Hello'}, template=tmpl, fragment=True)
    self.assertEqual('<html lang="en">\n  <head>\n    <title>Test</title>\n  </head>\n  <body>\n    <h1>Test</h1>\n    <p>Hello</p>\n  </body>\n</html>', output)