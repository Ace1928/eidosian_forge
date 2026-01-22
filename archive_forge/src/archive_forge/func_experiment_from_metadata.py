import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def experiment_from_metadata(self, ctx, experiment_id, hparams_run_to_tag_to_content, data_provider_hparams):
    """Returns the experiment proto defining the experiment.

        This method first attempts to find a metadata.EXPERIMENT_TAG tag and
        retrieve the associated proto.

        If no such tag is found, the method will attempt to build a minimal
        experiment proto by scanning for all metadata.SESSION_START_INFO_TAG
        tags (to compute the hparam_infos field of the experiment) and for all
        scalar tags (to compute the metric_infos field of the experiment).

        If no metadata.EXPERIMENT_TAG nor metadata.SESSION_START_INFO_TAG tags
        are found, then will build an experiment proto using the results from
        DataProvider.list_hyperparameters().

        Args:
          experiment_id: String, from `plugin_util.experiment_id`.
          hparams_run_to_tag_to_content: The output from an hparams_metadata()
            call. A dict `d` such that `d[run][tag]` is a `bytes` value with the
            summary metadata content for the keyed time series.
          data_provider_hparams: The ouput from an hparams_from_data_provider()
            call, corresponding to DataProvider.list_hyperparameters().
            A Collection[provider.Hyperparameter].

        Returns:
          The experiment proto. If no data is found for an experiment proto to
          be built, returns an entirely empty experiment.
        """
    experiment = self._find_experiment_tag(hparams_run_to_tag_to_content)
    if experiment:
        return experiment
    experiment_from_runs = self._compute_experiment_from_runs(ctx, experiment_id, hparams_run_to_tag_to_content)
    if experiment_from_runs:
        return experiment_from_runs
    experiment_from_data_provider_hparams = self._experiment_from_data_provider_hparams(data_provider_hparams)
    return experiment_from_data_provider_hparams if experiment_from_data_provider_hparams else api_pb2.Experiment()