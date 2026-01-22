from typing import Callable, List, Optional, Union
from urllib.parse import urlparse
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available
def image_tokens(last_was_image):
    if last_was_image:
        return image_token + fake_token
    else:
        return fake_token + image_token + fake_token