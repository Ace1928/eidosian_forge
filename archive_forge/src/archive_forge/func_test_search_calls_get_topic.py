import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_search_calls_get_topic(self):
    """Searching should call get_topics in all indexes in order."""
    calls = []

    class RecordingIndex:

        def __init__(self, name):
            self.prefix = name

        def get_topics(self, topic):
            calls.append(('get_topics', self.prefix, topic))
            return ['something']
    index = help.HelpIndices()
    index.search_path = [RecordingIndex('1'), RecordingIndex('2')]
    index.search(None)
    self.assertEqual([('get_topics', '1', None), ('get_topics', '2', None)], calls)
    del calls[:]
    index.search('bar')
    self.assertEqual([('get_topics', '1', 'bar'), ('get_topics', '2', 'bar')], calls)