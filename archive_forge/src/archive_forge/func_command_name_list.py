import textwrap
import breezy
import breezy.commands
import breezy.help
import breezy.help_topics
from breezy.doc_generate import get_autodoc_datetime
from breezy.plugin import load_plugins
def command_name_list():
    """Builds a list of command names from breezy"""
    command_names = breezy.commands.builtin_command_names()
    for cmdname in breezy.commands.plugin_command_names():
        cmd_object = breezy.commands.get_cmd_object(cmdname)
        if PLUGINS_TO_DOCUMENT is None or cmd_object.plugin_name() in PLUGINS_TO_DOCUMENT:
            command_names.append(cmdname)
    command_names.sort()
    return command_names