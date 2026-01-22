from typing import Dict, List, Optional
import torch
import torch.optim._functional as F
from torch import Tensor
Similar to self.step, but operates on a single parameter and
        its gradient.
        