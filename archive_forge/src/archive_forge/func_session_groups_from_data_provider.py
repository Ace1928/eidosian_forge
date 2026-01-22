import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def session_groups_from_data_provider(self, ctx, experiment_id, filters, sort):
    """Calls DataProvider.read_hyperparameters() and returns the result."""
    return self._tb_context.data_provider.read_hyperparameters(ctx, experiment_ids=[experiment_id], filters=filters, sort=sort)