import contextlib
import io
import os
import shutil
import tempfile
import textwrap
import tokenize
import unittest
import unittest.mock as mock
from traits.api import Bool, HasTraits, Int, Property
from traits.testing.optional_dependencies import sphinx, requires_sphinx
@contextlib.contextmanager
def create_directive(self):
    """
        Helper function to create a a "directive" suitable
        for instantiating the TraitDocumenter with, along with resources
        to support that directive, and clean up the resources afterwards.

        Returns
        -------
        contextmanager
            A context manager that returns a DocumenterBridge instance.
        """
    with self.tmpdir() as tmpdir:
        conf_file = os.path.join(tmpdir, 'conf.py')
        with open(conf_file, 'w', encoding='utf-8') as f:
            f.write(CONF_PY)
        app = SphinxTestApp(srcdir=path(tmpdir))
        app.builder.env.app = app
        app.builder.env.temp_data['docname'] = 'dummy'
        kwds = {}
        state = mock.Mock()
        state.document.settings.tab_width = 8
        kwds['state'] = state
        yield DocumenterBridge(app.env, LoggingReporter(''), Options(), 1, **kwds)