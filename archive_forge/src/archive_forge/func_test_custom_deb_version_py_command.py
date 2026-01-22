from testtools import content
from pbr.tests import base
def test_custom_deb_version_py_command(self):
    """Test custom deb_version command."""
    stdout, stderr, return_code = self.run_setup('deb_version')
    self.addDetail('stdout', content.text_content(stdout))
    self.addDetail('stderr', content.text_content(stderr))
    self.assertIn('Extracting deb version', stdout)
    self.assertEqual(0, return_code)