from . import commands as _mod_commands
from . import errors, help_topics, osutils, plugin, ui, utextwrap
def help_commands(outfile=None):
    """List all commands"""
    if outfile is None:
        outfile = ui.ui_factory.make_output_stream()
    outfile.write(_help_commands_to_text('commands'))