import torch
from torch.ao.quantization.observer import ObserverBase
@torch.jit.export
def get_batch_to_epoch_ratio(self):
    epoch_activation_range = self.epoch_activation_max - self.epoch_activation_min
    if epoch_activation_range == torch.tensor(float(0)):
        raise ValueError('Range for Epoch is 0')
    elif epoch_activation_range == torch.tensor(float('inf')):
        raise ValueError('No data has been run through observer or infinity value present')
    else:
        return self.average_batch_activation_range / epoch_activation_range