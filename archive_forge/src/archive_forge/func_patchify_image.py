import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def patchify_image(self, image: 'torch.Tensor', patch_size: Optional[Dict[str, int]]=None) -> 'torch.Tensor':
    """
        Convert an image into a tensor of patches.

        Args:
            image (`torch.Tensor`):
                Image to convert. Shape: [batch, channels, height, width]
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
    requires_backends(self, ['torch'])
    patch_size = patch_size if patch_size is not None else self.patch_size
    patch_height, patch_width = (patch_size['height'], patch_size['width'])
    batch_size, channels, _, _ = image.shape
    unfolded_along_height = image.unfold(2, patch_height, patch_height)
    patches = unfolded_along_height.unfold(3, patch_width, patch_width)
    patches = patches.contiguous()
    patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
    patches = patches.permute(0, 2, 3, 4, 1)
    patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
    return patches