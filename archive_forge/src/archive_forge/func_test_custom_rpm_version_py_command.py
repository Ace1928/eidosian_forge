from testtools import content
from pbr.tests import base
def test_custom_rpm_version_py_command(self):
    """Test custom rpm_version command."""
    stdout, stderr, return_code = self.run_setup('rpm_version')
    self.addDetail('stdout', content.text_content(stdout))
    self.addDetail('stderr', content.text_content(stderr))
    self.assertIn('Extracting rpm version', stdout)
    self.assertEqual(0, return_code)