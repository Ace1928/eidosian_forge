import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_render_with_doctype(self):
    plugin = MarkupTemplateEnginePlugin(options={'genshi.default_doctype': 'html-strict'})
    tmpl = plugin.load_template(PACKAGE + '.templates.test')
    output = plugin.render({'message': 'Hello'}, template=tmpl)
    self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html lang="en">\n  <head>\n    <title>Test</title>\n  </head>\n  <body>\n    <h1>Test</h1>\n    <p>Hello</p>\n  </body>\n</html>', output)