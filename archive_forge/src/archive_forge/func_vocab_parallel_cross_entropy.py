import torch
from .initialize import get_model_parallel_group, get_model_parallel_rank, get_model_parallel_world_size
from .utils import VocabUtility
def vocab_parallel_cross_entropy(vocab_parallel_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)