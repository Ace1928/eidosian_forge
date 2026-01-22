from collections import defaultdict
from typing import Any, Dict
import torch
from torch.utils._python_dispatch import TorchDispatchMode
def get_comm_counts(self) -> Dict[Any, int]:
    """Returns the communication counts as a dictionary.

        Returns:
            Dict[Any, int]: The communication counts as a dictionary.
        """
    return self.comm_counts