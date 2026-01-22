from typing import List, Optional
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import to_numpy

        This method strips any padding from the audio values to return a list of numpy audio arrays.
        