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
def scalars_multirun_impl(self, ctx, tag, runs, experiment):
    """Result of the form `(body, mime_type)`."""
    all_scalars = self._data_provider.read_scalars(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=self._downsample_to, run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]))
    body = {run: [(x.wall_time, x.step, x.value) for x in run_data[tag]] for run, run_data in all_scalars.items()}
    return (body, 'application/json')