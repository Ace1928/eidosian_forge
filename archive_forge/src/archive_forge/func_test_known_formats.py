from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_known_formats(self):
    known_formats = self.prober_cls.known_formats()
    self.assertIsInstance(known_formats, list)
    for format in known_formats:
        self.assertIsInstance(format, controldir.ControlDirFormat, repr(format))