from heat.engine.clients import progress
from heat.tests import common
def test_extra_all_defaults(self):
    prg = progress.ServerUpdateProgress(self.server_id, self.handler)
    self._assert_common(prg)
    self.assertEqual((self.server_id,), prg.handler_args)
    self.assertEqual((self.server_id,), prg.checker_args)
    self.assertEqual({}, prg.handler_kwargs)
    self.assertEqual({}, prg.checker_kwargs)