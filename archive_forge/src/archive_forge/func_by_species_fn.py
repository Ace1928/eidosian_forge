import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
@pn.depends(by_species.param.value, color.param.value)
def by_species_fn(by_species, color):
    return 'species' if by_species else color