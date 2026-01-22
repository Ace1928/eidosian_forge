from keras.src import initializers
from keras.src import losses
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.metrics.metric import Metric
from keras.src.saving import serialization_lib
def reduce_to_samplewise_values(values, sample_weight, reduce_fn, dtype):
    mask = getattr(values, '_keras_mask', None)
    values = ops.cast(values, dtype=dtype)
    if sample_weight is not None:
        sample_weight = ops.cast(sample_weight, dtype=dtype)
        if mask is not None:
            sample_weight = losses.loss.apply_mask(sample_weight, mask, dtype=dtype, reduction='sum')
        values, sample_weight = losses.loss.squeeze_or_expand_to_same_rank(values, sample_weight)
        weight_ndim = len(sample_weight.shape)
        values_ndim = len(values.shape)
        if values_ndim > weight_ndim:
            values = reduce_fn(values, axis=list(range(weight_ndim, values_ndim)))
        values = values * sample_weight
        if values_ndim > 1:
            sample_weight = reduce_fn(sample_weight, axis=list(range(1, weight_ndim)))
    values_ndim = len(values.shape)
    if values_ndim > 1:
        values = reduce_fn(values, axis=list(range(1, values_ndim)))
        return (values, sample_weight)
    return (values, sample_weight)