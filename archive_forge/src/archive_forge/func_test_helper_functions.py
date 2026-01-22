import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_helper_functions(self):
    plugin = TextTemplateEnginePlugin()
    tmpl = plugin.load_template(PACKAGE + '.templates.functions')
    output = plugin.render({}, template=tmpl)
    self.assertEqual('False\nbar\n', output)