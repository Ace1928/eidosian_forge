from typing import TYPE_CHECKING, Tuple
def bloom_convert_to_standard_cache(past_key_value: Tuple[Tuple['torch.Tensor', 'torch.Tensor']], batch_size: int) -> Tuple[Tuple['torch.Tensor', 'torch.Tensor']]:
    """
    Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
    num_heads, ...]))
    """
    batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
    num_heads = batch_size_times_num_heads // batch_size
    return tuple(((layer_past[0].view(batch_size, num_heads, head_dim, seq_length), layer_past[1].view(batch_size, num_heads, seq_length, head_dim)) for layer_past in past_key_value))