from ... import config, tests
from .. import script
from .. import test_config as _t_config
def test_multiline_value_only(self):
    self.breezy_config.set_user_option('multiline', '1\n2\n')
    script.run_script(self, '            $ brz config -d tree multiline\n            """1\n            2\n            """\n            ')