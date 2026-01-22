from __future__ import annotations
from typing import TYPE_CHECKING
import numba
import numpy as np

Numba 1D min/max kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
