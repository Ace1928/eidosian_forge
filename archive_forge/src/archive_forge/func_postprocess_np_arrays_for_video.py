import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import wandb
from wandb.util import get_module
def postprocess_np_arrays_for_video(images: List['np_array'], normalize: Optional[bool]=False) -> 'np_array':
    np = get_module('numpy', required='Please ensure NumPy is installed. You can run `pip install numpy` to install it.')
    images = [(img * 255).astype('uint8') for img in images] if normalize else images
    return np.transpose(np.stack(images, axis=0), axes=(0, 3, 1, 2))