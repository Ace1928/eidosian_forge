from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from ._utils import array_kind
from ._utils.registry import Registry
from .exceptions import PlotnineError, PlotnineWarning
from .facets import facet_grid, facet_null, facet_wrap
from .facets.facet_grid import parse_grid_facets_old
from .facets.facet_wrap import parse_wrap_facets_old
from .ggplot import ggplot
from .labels import labs
from .mapping.aes import ALL_AESTHETICS, SCALED_AESTHETICS, aes
from .scales import lims, scale_x_log10, scale_y_log10
from .themes import theme

        Replace all occurences of 'auto' in with str2
        