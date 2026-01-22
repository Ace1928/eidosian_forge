from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING, Dict, Any
import numpy as np
import pandas as pd
from cirq import sim, value
Helper function for simulating a given (circuit, cycle_depth).