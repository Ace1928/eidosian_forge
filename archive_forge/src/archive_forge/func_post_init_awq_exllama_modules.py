from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import (
def post_init_awq_exllama_modules(model, exllama_config):
    """
    Runs post init for Exllama layers which performs:
        - Weights unpacking, reordering and repacking
        - Devices scratch space allocation
    """
    if exllama_config['version'] == ExllamaVersion.ONE:
        from awq.modules.linear.exllama import exllama_post_init
        model = exllama_post_init(model)
    elif exllama_config['version'] == ExllamaVersion.TWO:
        from awq.modules.linear.exllamav2 import exllamav2_post_init
        model = exllamav2_post_init(model, max_input_len=exllama_config['max_input_len'], max_batch_size=exllama_config['max_batch_size'])
    else:
        raise ValueError(f'Unrecognized Exllama version: {exllama_config['version']}')
    return model