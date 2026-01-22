from pathlib import Path
import torch
from ...utils import is_npu_available, is_xpu_available
from .config_args import ClusterConfig, default_json_config_file
from .config_utils import SubcommandHelpFormatter
def write_basic_config(mixed_precision='no', save_location: str=default_json_config_file, use_xpu: bool=False):
    """
    Creates and saves a basic cluster config to be used on a local machine with potentially multiple GPUs. Will also
    set CPU if it is a CPU-only machine.

    Args:
        mixed_precision (`str`, *optional*, defaults to "no"):
            Mixed Precision to use. Should be one of "no", "fp16", or "bf16"
        save_location (`str`, *optional*, defaults to `default_json_config_file`):
            Optional custom save location. Should be passed to `--config_file` when using `accelerate launch`. Default
            location is inside the huggingface cache folder (`~/.cache/huggingface`) but can be overriden by setting
            the `HF_HOME` environmental variable, followed by `accelerate/default_config.yaml`.
        use_xpu (`bool`, *optional*, defaults to `False`):
            Whether to use XPU if available.
    """
    path = Path(save_location)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f'Configuration already exists at {save_location}, will not override. Run `accelerate config` manually or pass a different `save_location`.')
        return False
    mixed_precision = mixed_precision.lower()
    if mixed_precision not in ['no', 'fp16', 'bf16', 'fp8']:
        raise ValueError(f"`mixed_precision` should be one of 'no', 'fp16', 'bf16', or 'fp8'. Received {mixed_precision}")
    config = {'compute_environment': 'LOCAL_MACHINE', 'mixed_precision': mixed_precision}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        config['num_processes'] = num_gpus
        config['use_cpu'] = False
        if num_gpus > 1:
            config['distributed_type'] = 'MULTI_GPU'
        else:
            config['distributed_type'] = 'NO'
    elif is_xpu_available() and use_xpu:
        num_xpus = torch.xpu.device_count()
        config['num_processes'] = num_xpus
        config['use_cpu'] = False
        if num_xpus > 1:
            config['distributed_type'] = 'MULTI_XPU'
        else:
            config['distributed_type'] = 'NO'
    elif is_npu_available():
        num_npus = torch.npu.device_count()
        config['num_processes'] = num_npus
        config['use_cpu'] = False
        if num_npus > 1:
            config['distributed_type'] = 'MULTI_NPU'
        else:
            config['distributed_type'] = 'NO'
    else:
        num_xpus = 0
        config['use_cpu'] = True
        config['num_processes'] = 1
        config['distributed_type'] = 'NO'
    config['debug'] = False
    config = ClusterConfig(**config)
    config.to_json_file(path)
    return path