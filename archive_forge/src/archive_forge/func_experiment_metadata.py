import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def experiment_metadata(self, ctx=None, *, experiment_id):
    self._validate_context(ctx)
    self._validate_experiment_id(experiment_id)
    return provider.ExperimentMetadata(data_location=self._logdir)