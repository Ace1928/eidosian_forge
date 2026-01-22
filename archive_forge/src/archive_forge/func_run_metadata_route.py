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
def run_metadata_route(self, request):
    """Given a tag and a run, return the session.run() metadata."""
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    tag = request.args.get('tag')
    run = request.args.get('run')
    if tag is None:
        return http_util.Respond(request, 'query parameter "tag" is required', 'text/plain', 400)
    if run is None:
        return http_util.Respond(request, 'query parameter "run" is required', 'text/plain', 400)
    body, mime_type = self.run_metadata_impl(ctx, experiment, run, tag)
    return http_util.Respond(request, body, mime_type)