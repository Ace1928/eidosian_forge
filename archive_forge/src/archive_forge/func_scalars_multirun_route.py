import csv
import io
import werkzeug.exceptions
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata
@wrappers.Request.application
def scalars_multirun_route(self, request):
    """Given a tag and list of runs, return dict of ScalarEvent arrays."""
    if request.method != 'POST':
        raise werkzeug.exceptions.MethodNotAllowed(['POST'])
    tags = request.form.getlist('tag')
    runs = request.form.getlist('runs')
    if len(tags) != 1:
        raise errors.InvalidArgumentError('tag must be specified exactly once')
    tag = tags[0]
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    body, mime_type = self.scalars_multirun_impl(ctx, tag, runs, experiment)
    return http_util.Respond(request, body, mime_type)