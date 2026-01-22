import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_search_for_unknown_topic_raises(self):
    """Searching for an unknown topic should raise NoHelpTopic."""
    indices = help.HelpIndices()
    indices.search_path = []
    error = self.assertRaises(help.NoHelpTopic, indices.search, 'foo')
    self.assertEqual('foo', error.topic)