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
def new_v2(*args, **kwargs):
    cbks = kwargs.get('callbacks', [])
    val_data = kwargs.get('validation_data')
    if val_data:
        for cbk in cbks:
            set_wandb_attrs(cbk, val_data)
    return old_v2(*args, **kwargs)