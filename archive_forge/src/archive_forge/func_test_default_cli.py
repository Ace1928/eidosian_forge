import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def test_default_cli(self):
    filename = self._write_opt_to_tmp_file('DEFAULT', 'unknown_opt', self.id())
    self.conf(['--config-file', filename])
    loc = self.conf.get_location('cli_opt')
    self.assertEqual(cfg.Locations.opt_default, loc.location)