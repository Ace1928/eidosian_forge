import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def read_scalars(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
    del experiment_id, plugin_name, downsample, run_tag_filter
    raise TypeError("Debugger V2 DataProvider doesn't support scalars.")