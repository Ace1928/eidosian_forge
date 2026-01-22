from typing import Any, Optional, TYPE_CHECKING
import warnings
import pandas as pd
import sympy
from matplotlib import pyplot as plt
import numpy as np
from cirq import circuits, ops, study, value, _import
from cirq._compat import proper_repr
Text output in Jupyter.