import os
import tempfile
from tempfile import TemporaryDirectory
from traitlets import Unicode
from IPython.core.application import BaseIPythonApplication
from IPython.testing import decorators as dec
def test_cli_priority():
    with TemporaryDirectory() as td:

        class TestApp(BaseIPythonApplication):
            test = Unicode().tag(config=True)
        with open(os.path.join(td, 'ipython_config.py'), 'w', encoding='utf-8') as f:
            f.write("c.TestApp.test = 'config file'")
        app = TestApp()
        app.initialize(['--profile-dir', td])
        assert app.test == 'config file'
        app = TestApp()
        app.initialize(['--profile-dir', td, '--TestApp.test=cli'])
        assert app.test == 'cli'