import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
def update_node(self, index, new_index, offset_length):
    self._distance[new_index] = self._distance[index] + 1