from collections import OrderedDict, deque
import numpy as np
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_tf
def get_flat(self):
    """Gets the weights and returns them as a flat array.

        Returns:
            1D Array containing the flattened weights.
        """
    if not self.sess:
        return np.concatenate([v.numpy().flatten() for v in self.variables.values()])
    return np.concatenate([v.eval(session=self.sess).flatten() for v in self.variables.values()])