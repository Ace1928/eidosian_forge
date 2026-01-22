from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def then_data(self, content):
    self.assertIn("  cmds='image server'\n", content)
    self.assertIn("  cmds_image='create'\n", content)
    self.assertIn("  cmds_image_create='--eolus'\n", content)
    self.assertIn("  cmds_server='meta ssh'\n", content)
    self.assertIn("  cmds_server_meta_delete='--wilson'\n", content)
    self.assertIn("  cmds_server_ssh='--sunlight'\n", content)