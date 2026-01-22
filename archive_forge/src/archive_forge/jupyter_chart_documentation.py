import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection

        Internal callback function that updates the JupyterChart's public
        selections traitlet in response to changes that the JavaScript logic
        makes to the internal _selections traitlet.
        