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
@wrappers.Request.application
def info_route(self, request):
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    info = self.info_impl(ctx, experiment)
    return http_util.Respond(request, info, 'application/json')