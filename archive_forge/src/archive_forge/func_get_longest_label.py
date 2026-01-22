from __future__ import unicode_literals
from pybtex.plugin import Plugin
from pybtex.textutils import width
def get_longest_label(self, formatted_entries):
    labels = (entry.label for entry in formatted_entries)
    return max(labels, key=width)