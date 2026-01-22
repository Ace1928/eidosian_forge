import warnings
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineWarning
from .smoothers import predictdf
from .stat import stat

        Overide to modify data before compute_layer is called
        