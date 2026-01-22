import jax
import numpy as np
from keras.src.utils import jax_utils
def process_id():
    """Return the current process ID for the distribution setting."""
    return jax.process_index()