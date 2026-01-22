from collections import namedtuple
import numpy as np
import pytest
from ...data.io_numpyro import from_numpyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
Generate predictions for predictions_params