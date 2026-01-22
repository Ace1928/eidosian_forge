import time
import tensorflow as tf
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def experiment_pb(hparam_infos, metric_infos, user='', description='', time_created_secs=None):
    """Creates a summary that defines a hyperparameter-tuning experiment.

    Args:
      hparam_infos: Array of api_pb2.HParamInfo messages. Describes the
          hyperparameters used in the experiment.
      metric_infos: Array of api_pb2.MetricInfo messages. Describes the metrics
          used in the experiment. See the documentation at the top of this file
          for how to populate this.
      user: String. An id for the user running the experiment
      description: String. A description for the experiment. May contain markdown.
      time_created_secs: float. The time the experiment is created in seconds
      since the UNIX epoch. If None uses the current time.

    Returns:
      A summary protobuffer containing the experiment definition.
    """
    if time_created_secs is None:
        time_created_secs = time.time()
    experiment = api_pb2.Experiment(description=description, user=user, time_created_secs=time_created_secs, hparam_infos=hparam_infos, metric_infos=metric_infos)
    return _summary(metadata.EXPERIMENT_TAG, plugin_data_pb2.HParamsPluginData(experiment=experiment))