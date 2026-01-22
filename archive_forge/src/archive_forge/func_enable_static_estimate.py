import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def enable_static_estimate(self):
    """Enable static estimates of quantization parameters.

        Enables static observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
    self.toggle_qparam_learning(enabled=False).toggle_fake_quant(enabled=True).toggle_observer_update(enabled=True)