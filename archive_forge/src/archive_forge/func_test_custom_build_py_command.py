from testtools import content
from pbr.tests import base
def test_custom_build_py_command(self):
    """Test custom build_py command.

        Test that a custom subclass of the build_py command runs when listed in
        the commands [global] option, rather than the normal build command.
        """
    stdout, stderr, return_code = self.run_setup('build_py')
    self.addDetail('stdout', content.text_content(stdout))
    self.addDetail('stderr', content.text_content(stderr))
    self.assertIn('Running custom build_py command.', stdout)
    self.assertEqual(0, return_code)