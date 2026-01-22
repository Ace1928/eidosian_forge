import torch.distributed
from accelerate.test_utils import require_huggingface_suite
from accelerate.utils import is_transformers_available
@require_huggingface_suite
def init_torch_dist_then_launch_deepspeed():
    torch.distributed.init_process_group(backend='nccl')
    deepspeed_config = {'zero_optimization': {'stage': 3}, 'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto'}
    train_args = TrainingArguments(output_dir='./', deepspeed=deepspeed_config)
    model = AutoModel.from_pretrained(GPT2_TINY)
    assert train_args is not None
    assert model is not None