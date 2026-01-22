import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
@dist_init
def test_load_script_module_with_pickled_rref(self):
    dst_name = worker_name((self.rank + 1) % self.world_size)
    m1 = MyScriptModuleWithRRefs(dst_name)
    m2 = MyScriptModuleWithRRefs(dst_name)
    f = io.BytesIO()
    rpc._enable_jit_rref_pickle()
    torch.jit.save(m1, f)
    rpc._disable_jit_rref_pickle()
    out1 = rpc.rpc_sync(dst_name, load_script_module_with_pickled_rref, args=(f.getvalue(),))
    out2 = m2()
    self.assertEqual(out1, out2)