import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def new_arrays(*args, **kwargs):
    cbks = kwargs.get('callbacks', [])
    val_inputs = kwargs.get('val_inputs')
    val_targets = kwargs.get('val_targets')
    if val_inputs and val_targets:
        for cbk in cbks:
            set_wandb_attrs(cbk, (val_inputs[0], val_targets[0]))
    return old_arrays(*args, **kwargs)