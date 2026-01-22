import torch.nn as nn
from torch._functorch.utils import exposed_in
def batch_norm_without_running_stats(module: nn.Module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        module.track_running_stats = False