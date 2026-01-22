import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def hparams_metadata(self, ctx, experiment_id, run_tag_filter=None):
    """Reads summary metadata for all hparams time series.

        Args:
          experiment_id: String, from `plugin_util.experiment_id`.
          run_tag_filter: Optional `data.provider.RunTagFilter`, with
            the semantics as in `list_tensors`.

        Returns:
          A dict `d` such that `d[run][tag]` is a `bytes` value with the
          summary metadata content for the keyed time series.
        """
    return self._convert_plugin_metadata(self._tb_context.data_provider.list_tensors(ctx, experiment_id=experiment_id, plugin_name=metadata.PLUGIN_NAME, run_tag_filter=run_tag_filter))