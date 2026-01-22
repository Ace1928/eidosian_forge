import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
Process images for model input. In particular, variable-sized images are handled here.

        Args:
            image_input (`torch.Tensor` of shape [batch_size, subsequence_size, num_channels, height, width]):
                Tensor of images padded to model input size.
            image_present (`torch.Tensor` of shape [batch_size, subsequence_size, num_images]):
                Tensor of 1s and 0s indicating whether an image is present.
            image_unpadded_h (`torch.Tensor` of shape [batch_size, subsequence_size]):
                Tensor of unpadded image heights.
            image_unpadded_w (`torch.Tensor` of shape [batch_size, subsequence_size]):
                Tensor of unpadded image widths.
            image_placeholder_id (int):
                The id of the image placeholder token. Comes from an associated tokenizer.
            image_newline_id (int):
                The id of the image newline token. Comes from an associated tokenizer.
            variable_sized (bool):
                Whether to process images as variable-sized.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Size of the patches.
        