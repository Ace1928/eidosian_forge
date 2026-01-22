import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def update_hparam_info(self, hparam_info):
    hparam_info.type = {int: api_pb2.DATA_TYPE_FLOAT64, float: api_pb2.DATA_TYPE_FLOAT64, bool: api_pb2.DATA_TYPE_BOOL, str: api_pb2.DATA_TYPE_STRING}[self._dtype]
    hparam_info.ClearField('domain_discrete')
    hparam_info.domain_discrete.extend(self._values)