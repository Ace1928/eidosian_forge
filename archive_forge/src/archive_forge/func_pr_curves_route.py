import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
@wrappers.Request.application
def pr_curves_route(self, request):
    """A route that returns a JSON mapping between runs and PR curve data.

        Returns:
          Given a tag and a comma-separated list of runs (both stored within GET
          parameters), fetches a JSON object that maps between run name and objects
          containing data required for PR curves for that run. Runs that either
          cannot be found or that lack tags will be excluded from the response.
        """
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    runs = request.args.getlist('run')
    if not runs:
        return http_util.Respond(request, 'No runs provided when fetching PR curve data', 400)
    tag = request.args.get('tag')
    if not tag:
        return http_util.Respond(request, 'No tag provided when fetching PR curve data', 400)
    try:
        response = http_util.Respond(request, self.pr_curves_impl(ctx, experiment, runs, tag), 'application/json')
    except ValueError as e:
        return http_util.Respond(request, str(e), 'text/plain', 400)
    return response