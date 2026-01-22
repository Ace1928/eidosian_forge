from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_included_in_known_formats(self):
    controldir.ControlDirFormat.register_prober(NotBzrDirProber)
    self.addCleanup(controldir.ControlDirFormat.unregister_prober, NotBzrDirProber)
    formats = controldir.ControlDirFormat.known_formats()
    self.assertIsInstance(formats, list)
    for format in formats:
        if isinstance(format, NotBzrDirFormat):
            break
    else:
        self.fail('No NotBzrDirFormat in %s' % formats)