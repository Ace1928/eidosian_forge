from typing import Optional
import ray.cloudpickle as ray_pickle
from ray._private.utils import binary_to_hex, hex_to_binary
from ray.data.preprocessor import Preprocessor
from ray.train._checkpoint import Checkpoint
def set_preprocessor(self, preprocessor: Preprocessor):
    """Store a preprocessor with the checkpoint."""
    self.update_metadata({PREPROCESSOR_KEY: binary_to_hex(ray_pickle.dumps(preprocessor))})