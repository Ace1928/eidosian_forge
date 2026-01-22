from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.util import tensor_util
def parse_experiment_plugin_data(content):
    """Returns the experiment from HParam's
    SummaryMetadata.plugin_data.content.

    Raises HParamsError if the content doesn't have 'experiment' set or
    this file is incompatible with the version of the metadata stored.

    Args:
      content: The SummaryMetadata.plugin_data.content to use.
    """
    return _parse_plugin_data_as(content, 'experiment')