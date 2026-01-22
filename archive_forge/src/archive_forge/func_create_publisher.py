import codecs
import warnings
from typing import TYPE_CHECKING, Any, List, Type
import docutils
from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import Values
from docutils.io import FileInput, Input, NullOutput
from docutils.parsers import Parser
from docutils.parsers.rst import Parser as RSTParser
from docutils.readers import standalone
from docutils.transforms import Transform
from docutils.transforms.references import DanglingReferences
from docutils.writers import UnfilteredWriter
from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.transforms import (AutoIndexUpgrader, DoctreeReadEvent, FigureAligner,
from sphinx.transforms.i18n import (Locale, PreserveTranslatableMessages,
from sphinx.transforms.references import SphinxDomains
from sphinx.util import UnicodeDecodeErrorHandler, get_filetype, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.versioning import UIDTransform
def create_publisher(app: 'Sphinx', filetype: str) -> Publisher:
    reader = SphinxStandaloneReader()
    reader.setup(app)
    parser = app.registry.create_source_parser(app, filetype)
    if parser.__class__.__name__ == 'CommonMarkParser' and parser.settings_spec == ():
        from docutils.parsers.rst import Parser as RSTParser
        parser.settings_spec = RSTParser.settings_spec
    pub = Publisher(reader=reader, parser=parser, writer=SphinxDummyWriter(), source_class=SphinxFileInput, destination=NullOutput())
    defaults = {'traceback': True, **app.env.settings}
    if docutils.__version_info__[:2] >= (0, 19):
        pub.get_settings(**defaults)
    else:
        pub.settings = pub.setup_option_parser(**defaults).get_default_values()
    return pub