import textwrap
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
from breezy.doc_generate import get_autodoc_datetime
from breezy.plugin import load_plugins
def environment_variables():
    yield '.SH "ENVIRONMENT"\n'
    from breezy.help_topics import known_env_variables
    for k, desc in known_env_variables:
        yield '.TP\n'
        yield ('.I "%s"\n' % k)
        yield (man_escape(desc) + '\n')