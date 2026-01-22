import warnings
from typing import List, Optional, Union
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType
def get_overflowing_images(self, images, overflow_to_sample_mapping):
    images_with_overflow = []
    for sample_idx in overflow_to_sample_mapping:
        images_with_overflow.append(images[sample_idx])
    if len(images_with_overflow) != len(overflow_to_sample_mapping):
        raise ValueError(f'Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}')
    return images_with_overflow