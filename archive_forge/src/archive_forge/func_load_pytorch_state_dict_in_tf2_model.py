import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def load_pytorch_state_dict_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None, ignore_mismatched_sizes=False):
    """Load a pytorch state_dict in a TF 2.0 model. pt_state_dict can be either an actual dict or a lazy-loading
    safetensors archive created with the safe_open() function."""
    import tensorflow as tf
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if _prefix is None:
        _prefix = ''
    if tf_inputs:
        with tf.name_scope(_prefix):
            tf_model(tf_inputs, training=False)
    tf_keys_to_pt_keys = {}
    for key in pt_state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if 'running_var' in key:
            new_key = key.replace('running_var', 'moving_variance')
        if 'running_mean' in key:
            new_key = key.replace('running_mean', 'moving_mean')
        key_components = key.split('.')
        name = None
        if key_components[-3::2] == ['parametrizations', 'original0']:
            name = key_components[-2] + '_g'
        elif key_components[-3::2] == ['parametrizations', 'original1']:
            name = key_components[-2] + '_v'
        if name is not None:
            key_components = key_components[:-3] + [name]
            new_key = '.'.join(key_components)
        if new_key is None:
            new_key = key
        tf_keys_to_pt_keys[new_key] = key
    start_prefix_to_remove = ''
    if not any((s.startswith(tf_model.base_model_prefix) for s in tf_keys_to_pt_keys.keys())):
        start_prefix_to_remove = tf_model.base_model_prefix + '.'
    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    all_pytorch_weights = set(tf_keys_to_pt_keys.keys())
    missing_keys = []
    mismatched_keys = []
    is_safetensor_archive = hasattr(pt_state_dict, 'get_tensor')
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(sw_name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=symbolic_weight.shape, name_scope=_prefix)
        if tf_to_pt_weight_rename is not None:
            aliases = tf_to_pt_weight_rename(name)
            for alias in aliases:
                if alias in tf_keys_to_pt_keys:
                    name = alias
                    break
            else:
                name = aliases[0]
        if name not in tf_keys_to_pt_keys:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                if any((re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing)):
                    continue
            raise AttributeError(f'{name} not found in PyTorch model')
        state_dict_name = tf_keys_to_pt_keys[name]
        if is_safetensor_archive:
            array = pt_state_dict.get_tensor(state_dict_name)
        else:
            array = pt_state_dict[state_dict_name]
        try:
            array = apply_transpose(transpose, array, symbolic_weight.shape)
        except tf.errors.InvalidArgumentError as e:
            if not ignore_mismatched_sizes:
                error_msg = str(e)
                error_msg += '\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method.'
                raise tf.errors.InvalidArgumentError(error_msg)
            else:
                mismatched_keys.append((name, array.shape, symbolic_weight.shape))
                continue
        tf_loaded_numel += tensor_size(array)
        symbolic_weight.assign(tf.cast(array, symbolic_weight.dtype))
        del array
        all_pytorch_weights.discard(name)
    logger.info(f'Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.')
    unexpected_keys = list(all_pytorch_weights)
    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    if len(unexpected_keys) > 0:
        logger.warning(f'Some weights of the PyTorch model were not used when initializing the TF 2.0 model {tf_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {tf_model.__class__.__name__} from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).')
    else:
        logger.warning(f'All PyTorch model weights were used when initializing {tf_model.__class__.__name__}.\n')
    if len(missing_keys) > 0:
        logger.warning(f'Some weights or buffers of the TF 2.0 model {tf_model.__class__.__name__} were not initialized from the PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    else:
        logger.warning(f'All the weights of {tf_model.__class__.__name__} were initialized from the PyTorch model.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {tf_model.__class__.__name__} for predictions without further training.')
    if len(mismatched_keys) > 0:
        mismatched_warning = '\n'.join([f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated' for key, shape1, shape2 in mismatched_keys])
        logger.warning(f'Some weights of {tf_model.__class__.__name__} were not initialized from the model checkpoint are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    if output_loading_info:
        loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'mismatched_keys': mismatched_keys}
        return (tf_model, loading_info)
    return tf_model