import numpy as np
import ray
import ray.experimental.array.remote as ra
@staticmethod
def compute_num_blocks(shape):
    return [int(np.ceil(1.0 * a / BLOCK_SIZE)) for a in shape]