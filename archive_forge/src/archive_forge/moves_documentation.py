from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast
import numpy as np
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default

    Divisive scaling on the value axis after aggregating within groups.

    Parameters
    ----------
    func : str or callable
        Function called on each group to define the comparison value.
    where : str
        Query string defining the subset used to define the comparison values.
    by : list of variables
        Variables used to define aggregation groups.
    percent : bool
        If True, multiply the result by 100.

    Examples
    --------
    .. include:: ../docstrings/objects.Norm.rst

    