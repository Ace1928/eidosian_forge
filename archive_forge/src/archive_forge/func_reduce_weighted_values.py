import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.utils.naming import auto_name
def reduce_weighted_values(values, sample_weight=None, mask=None, reduction='sum_over_batch_size', dtype=None):
    reduction = standardize_reduction(reduction)
    values = ops.convert_to_tensor(values, dtype=dtype)
    if sample_weight is not None:
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=dtype)
    if mask is not None:
        mask = ops.convert_to_tensor(mask, dtype=dtype)
    sample_weight = apply_mask(sample_weight, mask, dtype=values.dtype, reduction=reduction)
    if sample_weight is not None:
        sample_weight = ops.cast(sample_weight, values.dtype)
        values, sample_weight = squeeze_or_expand_to_same_rank(values, sample_weight)
        values = values * sample_weight
    loss = reduce_values(values, reduction)
    return loss