import jax
import numpy as np
from keras.src.utils import jax_utils
def num_processes():
    """Return the number of processes for the current distribution setting."""
    return jax.process_count()