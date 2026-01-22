import io
import math
from typing import Dict, Optional, Union
import numpy as np
from huggingface_hub import hf_hub_download
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
from ...utils import TensorType, is_torch_available, is_vision_available, logging
from ...utils.import_utils import requires_backends
def extract_flattened_patches(self, image: np.ndarray, max_patches: int, patch_size: dict, input_data_format: Optional[Union[str, ChannelDimension]]=None, **kwargs) -> np.ndarray:
    """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.

        Returns:
            result (`np.ndarray`):
                A sequence of `max_patches` flattened patches.
        """
    requires_backends(self.extract_flattened_patches, 'torch')
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)
    image = torch.from_numpy(image)
    patch_height, patch_width = (patch_size['height'], patch_size['width'])
    image_height, image_width = get_image_size(image, ChannelDimension.FIRST)
    scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
    num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    resized_height = max(num_feasible_rows * patch_height, 1)
    resized_width = max(num_feasible_cols * patch_width, 1)
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False, antialias=True).squeeze(0)
    patches = torch_extract_patches(image, patch_height, patch_width)
    patches_shape = patches.shape
    rows = patches_shape[1]
    columns = patches_shape[2]
    depth = patches_shape[3]
    patches = patches.reshape([rows * columns, depth])
    row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
    col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])
    row_ids += 1
    col_ids += 1
    row_ids = row_ids.to(torch.float32)
    col_ids = col_ids.to(torch.float32)
    result = torch.cat([row_ids, col_ids, patches], -1)
    result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - rows * columns]).float()
    result = to_numpy_array(result)
    return result