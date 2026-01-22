import logging
from collections import abc, defaultdict
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
from torch.distributed.distributed_c10d import ProcessGroup
def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
    if not self._enabled:
        return
    self._check_scale_growth_tracker('unscale_')
    optimizer_state = self._per_optimizer_states[id(optimizer)]
    if optimizer_state['stage'] is OptState.UNSCALED:
        raise RuntimeError('unscale_() has already been called on this optimizer since the last update().')
    elif optimizer_state['stage'] is OptState.STEPPED:
        raise RuntimeError('unscale_() is being called after step().')
    assert self._scale is not None
    inv_scale = self._scale.double().reciprocal().float()
    found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=self._scale.device)
    optimizer_state['found_inf_per_device'] = self._unscale_grads_(optimizer, inv_scale, found_inf, True)
    optimizer_state['stage'] = OptState.UNSCALED
    optimizer_state = self._per_optimizer_states[id(optimizer)]
    future_handles = []
    for v in optimizer_state['found_inf_per_device'].values():
        if v.device.type == 'cpu':
            v_on_cuda = v.cuda()
            future_handles.append(dist.all_reduce(v_on_cuda, async_op=True, group=self.process_group).get_future())
            v.copy_(v_on_cuda.cpu())
        else:
            future_handles.append(dist.all_reduce(v, async_op=True, group=self.process_group).get_future())
    if future_handles:
        torch.futures.wait_all(future_handles)