import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, List, Optional, Union
import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, create_repo
from packaging.version import parse
from . import IntervalStrategy, PreTrainedTokenizerBase
from .modelcard import TrainingSummary
from .modeling_tf_utils import keras

    Callback that will save and push the model to the Hub regularly. By default, it pushes once per epoch, but this can
    be changed with the `save_strategy` argument. Pushed models can be accessed like any other model on the hub, such
    as with the `from_pretrained` method.

    ```py
    from transformers.keras_callbacks import PushToHubCallback

    push_to_hub_callback = PushToHubCallback(
        output_dir="./model_save",
        tokenizer=tokenizer,
        hub_model_id="gpt5-7xlarge",
    )

    model.fit(train_dataset, callbacks=[push_to_hub_callback])
    ```

    Args:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written and synced with the
            repository on the Hub.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"epoch"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: Save is done at the end of training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`
        save_steps (`int`, *optional*):
            The number of steps between saves when using the "steps" `save_strategy`.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            The tokenizer used by the model. If supplied, will be uploaded to the repo alongside the weights.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local `output_dir`. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`.

            Will default to the name of `output_dir`.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to save full training checkpoints (including epoch and optimizer state) to allow training to be
            resumed. Only usable when `save_strategy` is `"epoch"`.
    