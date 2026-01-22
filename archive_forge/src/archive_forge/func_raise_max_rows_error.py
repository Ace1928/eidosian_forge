import json
import os
import random
import hashlib
import warnings
from typing import Union, MutableMapping, Optional, Dict, Sequence, TYPE_CHECKING, List
import pandas as pd
from toolz import curried
from typing import TypeVar
from ._importers import import_pyarrow_interchange
from .core import sanitize_dataframe, sanitize_arrow_table, DataFrameLike
from .core import sanitize_geo_interface
from .deprecation import AltairDeprecationWarning
from .plugin_registry import PluginRegistry
from typing import Protocol, TypedDict, Literal
def raise_max_rows_error():
    raise MaxRowsError(f'The number of rows in your dataset is greater than the maximum allowed ({max_rows}).\n\nTry enabling the VegaFusion data transformer which raises this limit by pre-evaluating data\ntransformations in Python.\n    >> import altair as alt\n    >> alt.data_transformers.enable("vegafusion")\n\nOr, see https://altair-viz.github.io/user_guide/large_datasets.html for additional information\non how to plot large datasets.')