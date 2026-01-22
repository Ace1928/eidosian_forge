import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=[], messages_to_delay={'SCRIPT_CALL': 1.5})
def test_rpc_script_timeout(self):
    next_rank = (self.rank + 1) % self.world_size
    dst_worker = worker_name(next_rank)
    expected_error = self.get_timeout_error_regex()
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rpc.rpc_sync(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)
    with self.assertRaisesRegex(RuntimeError, expected_error):
        fut.wait()
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
    fut.wait()
    rpc._set_rpc_timeout(0.001)
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
    with self.assertRaisesRegex(RuntimeError, expected_error):
        fut.wait()
    rpc._set_rpc_timeout(0.001)
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=0)
    fut.wait()
    rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)