from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_plugin_suffix():
    plugin = pybtex.plugin.find_plugin('pybtex.database.input', filename='test.bib')
    assert plugin is pybtex.database.input.bibtex.Parser