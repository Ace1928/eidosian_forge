import unittest
from functools import partial
from itertools import product
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import all_types_and, floating_types
from torch.testing._internal.common_utils import TEST_SCIPY, torch_to_numpy_dtype_dict
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import (
from torch.testing._internal.opinfo.utils import (
def reference_polygamma(x, n):
    result_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    if x.dtype == np.double:
        result_dtype = np.double
    return scipy.special.polygamma(n, x).astype(result_dtype)