from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick.caches import opencas
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.executor.Executor._execute')
def test_is_engine_ready(self, moc_exec):
    out_ready = 'type  id  disk  status  write policy  device\n        cache  1  /dev/nvme0n1  Running  wt  -'
    out_not_ready = 'type  id  disk  status  write policy  device'
    err = ''
    engine = opencas.OpenCASEngine(root_helper=None, opencas_cache_id=1)
    moc_exec.return_value = (out_ready, err)
    ret = engine.is_engine_ready()
    self.assertTrue(ret)
    moc_exec.return_value = (out_not_ready, err)
    ret = engine.is_engine_ready()
    self.assertFalse(ret)
    moc_exec.assert_has_calls([mock.call('casadm', '-L', run_as_root=True, root_helper=None)])