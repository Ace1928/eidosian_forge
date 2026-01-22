import torch
from torch.ao.quantization.observer import ObserverBase
@torch.jit.export
def reset_batch_and_epoch_values(self):
    device = self.max_val.device
    self.num_batches_tracked = 0
    self.average_batch_activation_range = torch.tensor(float(0), device=device)
    self.epoch_activation_min = torch.tensor(float('inf'), device=device)
    self.epoch_activation_max = torch.tensor(float('-inf'), device=device)
    self.min_val = torch.tensor([], device=device)
    self.max_val = torch.tensor([], device=device)
    self.average_percentile_ratio = torch.tensor([], device=device)
    self.percentile_batches_tracked = torch.tensor([], device=device)
    self.constant_channels = torch.tensor([], device=device)