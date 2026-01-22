import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_transform_without_load(self):
    plugin = TextTemplateEnginePlugin()
    stream = plugin.transform({'message': 'Hello'}, PACKAGE + '.templates.test')
    assert isinstance(stream, Stream)