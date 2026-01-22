import logging
from typing import List, Tuple
import torch
import torch.nn as nn
def scale_gradients(self, m: nn.Module, inputs: Tuple, outputs: Tuple) -> Tuple[torch.Tensor]:
    """
        Backward hook that is attached to the layers to scale the gradients.
        """
    scaled_up_grads = list()
    for idx in range(len(inputs)):
        if inputs[idx] is not None:
            if self.inputs_multiplier != 1.0 or self.outputs_multiplier != 1.0:
                logging.debug('layer = %s \t scale = %s \t scale_down = %s' % (self.layer_name, self.inputs_multiplier, self.outputs_multiplier))
            scaled_up_grads.append(inputs[idx].mul(self.inputs_multiplier * self.outputs_multiplier))
        else:
            logging.debug('next layer is None')
            scaled_up_grads.append(inputs[idx])
    return tuple(scaled_up_grads)