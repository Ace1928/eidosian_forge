import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def publish_cmdline(reader=None, reader_name='standalone', parser=None, parser_name='restructuredtext', writer=None, writer_name='pseudoxml', settings=None, settings_spec=None, settings_overrides=None, config_section=None, enable_exit_status=True, argv=None, usage=default_usage, description=default_description):
    """
    Set up & run a `Publisher` for command-line-based file I/O (input and
    output file paths taken automatically from the command line).  Return the
    encoded string output also.

    Parameters: see `publish_programmatically` for the remainder.

    - `argv`: Command-line argument list to use instead of ``sys.argv[1:]``.
    - `usage`: Usage string, output if there's a problem parsing the command
      line.
    - `description`: Program description, output for the "--help" option
      (along with command-line option descriptions).
    """
    pub = Publisher(reader, parser, writer, settings=settings)
    pub.set_components(reader_name, parser_name, writer_name)
    output = pub.publish(argv, usage, description, settings_spec, settings_overrides, config_section=config_section, enable_exit_status=enable_exit_status)
    return output