from __future__ import unicode_literals
import re
import pytest
import pybtex.database.input.bibtex
import pybtex.plugin
import pybtex.style.formatting.plain
def test_plugin_loader():
    """Check that all enumerated plugins can be imported."""
    for group in pybtex.plugin._DEFAULT_PLUGINS:
        for name in pybtex.plugin.enumerate_plugin_names(group):
            pybtex.plugin.find_plugin(group, name)