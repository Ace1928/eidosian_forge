from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_register_plugin_3():
    assert pybtex.plugin.register_plugin('pybtex.style.formatting.suffixes', '.woo', TestPlugin3)
    plugin = pybtex.plugin.find_plugin('pybtex.style.formatting', filename='test.woo')
    assert plugin is TestPlugin3