import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import metadata
from tensorboard.util import tb_logging
def data_plugin_names(self):
    return (metadata.PLUGIN_NAME, metadata.PLUGIN_NAME_RUN_METADATA, metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH, metadata.PLUGIN_NAME_KERAS_MODEL, metadata.PLUGIN_NAME_TAGGED_RUN_METADATA)