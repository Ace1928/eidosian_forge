import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def enable_static_observation(self):
    """Enable accumulation of data without updating quantization parameters.

        Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
    self.toggle_qparam_learning(enabled=False).toggle_fake_quant(enabled=False).toggle_observer_update(enabled=True)