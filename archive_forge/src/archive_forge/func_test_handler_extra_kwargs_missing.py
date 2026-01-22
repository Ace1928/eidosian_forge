from heat.engine.clients import progress
from heat.tests import common
def test_handler_extra_kwargs_missing(self):
    handler_extra = {'args': ()}
    prg = progress.ServerUpdateProgress(self.server_id, self.handler, handler_extra=handler_extra)
    self._assert_common(prg)
    self.assertEqual((self.server_id,), prg.handler_args)
    self.assertEqual((self.server_id,), prg.checker_args)
    self.assertEqual({}, prg.handler_kwargs)
    self.assertEqual({}, prg.checker_kwargs)