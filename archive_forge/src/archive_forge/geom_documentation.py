from __future__ import annotations
import typing
from abc import ABC
from copy import deepcopy
from itertools import chain, repeat
from .._utils import (
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError
from ..layer import layer
from ..mapping.aes import is_valid_aesthetic, rename_aesthetics
from ..mapping.evaluation import evaluate
from ..positions.position import position
from ..stats.stat import stat

        Calculate the size of key that would fit the layer contents

        Parameters
        ----------
        data :
            A row of the data plotted to this layer
        min_size :
            Initial size which should be expanded to fit the contents.
        lyr :
            Layer
        