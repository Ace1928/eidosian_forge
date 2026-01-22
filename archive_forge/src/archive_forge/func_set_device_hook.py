import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .modeling_base import PreTrainedModelWrapper
def set_device_hook(module, input, outputs):
    """
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
    new_output = ()
    for output in outputs:
        if isinstance(output, torch.Tensor):
            new_output += (output.to(lm_head_device),)
        else:
            new_output += (output,)
    return new_output