from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
def sync_shared_params(model: torch.nn.Module, process_group: ProcessGroup):
    pamams_shared = {name: p for name, p in model.named_parameters() if getattr(p, '_shared_params', False)}
    for _, p in sorted(pamams_shared.items()):
        with torch.no_grad():
            torch.distributed.broadcast(p, src=torch.distributed.get_global_rank(process_group, 0), group=process_group)