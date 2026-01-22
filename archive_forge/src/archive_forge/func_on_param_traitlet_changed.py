import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
def on_param_traitlet_changed(param_change):
    new_params = dict(self._params)
    new_params[param_change['name']] = param_change['new']
    self._params = new_params