import getpass
import os
from pathlib import Path
from tempfile import mkdtemp
def configure_jupyter_server(c):
    """Helper to configure the Jupyter Server for integration testing
    with Galata.

    By default the tests will be executed in the OS temporary folder. You
    can override that folder by setting the environment variable ``JUPYTERLAB_GALATA_ROOT_DIR``.

    .. warning::

        Never use this configuration in production as it will remove all security protections.
    """
    if getpass.getuser() == 'jovyan':
        c.ServerApp.ip = '0.0.0.0'
    c.ServerApp.port = 8888
    c.ServerApp.port_retries = 0
    c.ServerApp.open_browser = False
    c.LabServerApp.extra_labextensions_path = str(Path(__file__).parent)
    c.ServerApp.root_dir = os.environ.get('JUPYTERLAB_GALATA_ROOT_DIR', mkdtemp(prefix='galata-test-'))
    c.ServerApp.token = ''
    c.ServerApp.password = ''
    c.ServerApp.disable_check_xsrf = True
    c.LabApp.expose_app_in_browser = True