import textwrap
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
from breezy.doc_generate import get_autodoc_datetime
from breezy.plugin import load_plugins
def format_alias(params, alias, cmd_name):
    help = '.SS "brz %s"\n' % alias
    help += 'Alias for "{}", see "brz {}".\n'.format(cmd_name, cmd_name)
    return help