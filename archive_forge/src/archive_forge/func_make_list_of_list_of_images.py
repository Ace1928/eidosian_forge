import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def make_list_of_list_of_images(images: Union[List[List[ImageInput]], List[ImageInput], ImageInput]) -> List[List[ImageInput]]:
    if is_valid_image(images):
        return [[images]]
    if isinstance(images, list) and all((isinstance(image, list) for image in images)):
        return images
    if isinstance(images, list):
        return [make_list_of_images(image) for image in images]
    raise ValueError('images must be a list of list of images or a list of images or an image.')