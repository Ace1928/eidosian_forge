import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape

    Tests whether casting an element will faithfully copy data and
    parameters. Important to check for elements where data is not all
    held on .data attribute, e.g. Image bounds or Graph nodes and
    edgepaths.
    