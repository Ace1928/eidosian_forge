import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_default_search_path(self):
    """The default search path should include internal indexs."""
    indices = help.HelpIndices()
    self.assertEqual(4, len(indices.search_path))
    self.assertIsInstance(indices.search_path[0], help_topics.HelpTopicIndex)
    self.assertIsInstance(indices.search_path[1], commands.HelpCommandIndex)
    self.assertIsInstance(indices.search_path[2], plugin.PluginsHelpIndex)
    self.assertIsInstance(indices.search_path[3], help_topics.ConfigOptionHelpIndex)