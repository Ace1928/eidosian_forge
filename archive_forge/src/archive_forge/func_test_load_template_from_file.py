import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_load_template_from_file(self):
    plugin = TextTemplateEnginePlugin()
    tmpl = plugin.load_template(PACKAGE + '.templates.test')
    assert isinstance(tmpl, TextTemplate)
    self.assertEqual('test.txt', os.path.basename(tmpl.filename))