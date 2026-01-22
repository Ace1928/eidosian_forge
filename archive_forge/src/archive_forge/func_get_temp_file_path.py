import fixtures
from oslo_config import fixture as config
from oslotest import base as test_base
from oslo_service import _options
from oslo_service import sslutils
def get_temp_file_path(self, filename, root=None):
    """Returns an absolute path for a temporary file.

        If root is None, the file is created in default temporary directory. It
        also creates the directory if it's not initialized yet.

        If root is not None, the file is created inside the directory passed as
        root= argument.

        :param filename: filename
        :type filename: string
        :param root: temporary directory to create a new file in
        :type root: fixtures.TempDir
        :returns: absolute file path string
        """
    root = root or self.get_default_temp_dir()
    return root.join(filename)