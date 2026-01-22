from .. import config, debug, tests
def test_set_single_debug_flags_from_config(self):
    self.assertDebugFlags(['hpss'], b'debug_flags = hpss\n')