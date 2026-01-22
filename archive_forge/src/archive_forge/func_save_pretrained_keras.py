import collections.abc as collections
import json
import os
import warnings
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from huggingface_hub.utils import (
from .constants import CONFIG_NAME
from .hf_api import HfApi
from .utils import SoftTemporaryDirectory, logging, validate_hf_hub_args
def save_pretrained_keras(model, save_directory: Union[str, Path], config: Optional[Dict[str, Any]]=None, include_optimizer: bool=False, plot_model: bool=True, tags: Optional[Union[list, str]]=None, **model_save_kwargs):
    """
    Saves a Keras model to save_directory in SavedModel format. Use this if
    you're using the Functional or Sequential APIs.

    Args:
        model (`Keras.Model`):
            The [Keras
            model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
            you'd like to save. The model must be compiled and built.
        save_directory (`str` or `Path`):
            Specify directory in which you want to save the Keras model.
        config (`dict`, *optional*):
            Configuration object to be saved alongside the model weights.
        include_optimizer(`bool`, *optional*, defaults to `False`):
            Whether or not to include optimizer in serialization.
        plot_model (`bool`, *optional*, defaults to `True`):
            Setting this to `True` will plot the model and put it in the model
            card. Requires graphviz and pydot to be installed.
        tags (Union[`str`,`list`], *optional*):
            List of tags that are related to model or string of a single tag. See example tags
            [here](https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1).
        model_save_kwargs(`dict`, *optional*):
            model_save_kwargs will be passed to
            [`tf.keras.models.save_model()`](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model).
    """
    if is_tf_available():
        import tensorflow as tf
    else:
        raise ImportError('Called a Tensorflow-specific function but could not import it.')
    if not model.built:
        raise ValueError('Model should be built before trying to save')
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    if config:
        if not isinstance(config, dict):
            raise RuntimeError(f"Provided config to save_pretrained_keras should be a dict. Got: '{type(config)}'")
        with (save_directory / CONFIG_NAME).open('w') as f:
            json.dump(config, f)
    metadata = {}
    if isinstance(tags, list):
        metadata['tags'] = tags
    elif isinstance(tags, str):
        metadata['tags'] = [tags]
    task_name = model_save_kwargs.pop('task_name', None)
    if task_name is not None:
        warnings.warn('`task_name` input argument is deprecated. Pass `tags` instead.', FutureWarning)
        if 'tags' in metadata:
            metadata['tags'].append(task_name)
        else:
            metadata['tags'] = [task_name]
    if model.history is not None:
        if model.history.history != {}:
            path = save_directory / 'history.json'
            if path.exists():
                warnings.warn('`history.json` file already exists, it will be overwritten by the history of this version.', UserWarning)
            with path.open('w', encoding='utf-8') as f:
                json.dump(model.history.history, f, indent=2, sort_keys=True)
    _create_model_card(model, save_directory, plot_model, metadata)
    tf.keras.models.save_model(model, save_directory, include_optimizer=include_optimizer, **model_save_kwargs)