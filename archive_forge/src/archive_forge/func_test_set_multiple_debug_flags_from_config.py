from .. import config, debug, tests
def test_set_multiple_debug_flags_from_config(self):
    self.assertDebugFlags(['hpss', 'error'], b'debug_flags = hpss, error\n')