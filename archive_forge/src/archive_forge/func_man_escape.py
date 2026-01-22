import textwrap
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
from breezy.doc_generate import get_autodoc_datetime
from breezy.plugin import load_plugins
def man_escape(string):
    """Escapes strings for man page compatibility"""
    result = string.replace('\\', '\\\\')
    result = result.replace('`', "\\'")
    result = result.replace("'", '\\*(Aq')
    result = result.replace('-', '\\-')
    return result