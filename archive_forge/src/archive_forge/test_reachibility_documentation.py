import numpy as np
import pytest
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.utils._testing import (
Check that the computation preserve dtype thanks to fused types.