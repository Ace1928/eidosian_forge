import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_init_no_options(self):
    plugin = TextTemplateEnginePlugin()
    self.assertEqual(None, plugin.default_encoding)
    self.assertEqual('text', plugin.default_format)
    self.assertEqual([], plugin.loader.search_path)
    self.assertEqual(True, plugin.loader.auto_reload)
    self.assertEqual(25, plugin.loader._cache.capacity)