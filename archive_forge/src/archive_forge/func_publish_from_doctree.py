import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def publish_from_doctree(document, destination_path=None, writer=None, writer_name='pseudoxml', settings=None, settings_spec=None, settings_overrides=None, config_section=None, enable_exit_status=False):
    """
    Set up & run a `Publisher` to render from an existing document
    tree data structure, for programmatic use with string I/O.  Return
    the encoded string output.

    Note that document.settings is overridden; if you want to use the settings
    of the original `document`, pass settings=document.settings.

    Also, new document.transformer and document.reporter objects are
    generated.

    For encoded string output, be sure to set the 'output_encoding' setting to
    the desired encoding.  Set it to 'unicode' for unencoded Unicode string
    output.  Here's one way::

        publish_from_doctree(
            ..., settings_overrides={'output_encoding': 'unicode'})

    Parameters: `document` is a `docutils.nodes.document` object, an existing
    document tree.

    Other parameters: see `publish_programmatically`.
    """
    reader = docutils.readers.doctree.Reader(parser_name='null')
    pub = Publisher(reader, None, writer, source=io.DocTreeInput(document), destination_class=io.StringOutput, settings=settings)
    if not writer and writer_name:
        pub.set_writer(writer_name)
    pub.process_programmatic_settings(settings_spec, settings_overrides, config_section)
    pub.set_destination(None, destination_path)
    return pub.publish(enable_exit_status=enable_exit_status)