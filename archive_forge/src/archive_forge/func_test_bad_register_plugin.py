from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_bad_register_plugin():
    with pytest.raises(pybtex.plugin.PluginGroupNotFound):
        pybtex.plugin.register_plugin('pybtex.invalid.group', '__oops', TestPlugin1)
    with pytest.raises(pybtex.plugin.PluginGroupNotFound):
        pybtex.plugin.register_plugin('pybtex.invalid.group.suffixes', '.__oops', TestPlugin1)
    with pytest.raises(ValueError):
        pybtex.plugin.register_plugin('pybtex.style.formatting.suffixes', 'notasuffix', TestPlugin1)