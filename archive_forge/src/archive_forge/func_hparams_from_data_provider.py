import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def hparams_from_data_provider(self, ctx, experiment_id):
    """Calls DataProvider.list_hyperparameters() and returns the result."""
    return self._tb_context.data_provider.list_hyperparameters(ctx, experiment_ids=[experiment_id])