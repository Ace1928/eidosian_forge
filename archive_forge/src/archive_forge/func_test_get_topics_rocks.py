import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_get_topics_rocks(self):
    """Searching for 'rocks' returns the cmd_rocks command instance."""
    index = commands.HelpCommandIndex()
    topics = index.get_topics('rocks')
    self.assertEqual(1, len(topics))
    self.assertIsInstance(topics[0], builtins.cmd_rocks)