import textwrap
import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata
def text_impl(self, ctx, run, tag, experiment, enable_markdown):
    all_text = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=self._downsample_to, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]))
    text = all_text.get(run, {}).get(tag, None)
    if text is None:
        return []
    return [process_event(d.wall_time, d.step, d.numpy, enable_markdown) for d in text]