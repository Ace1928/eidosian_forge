from typing import Callable, List, Optional, Union
from urllib.parse import urlparse
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available
def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if ' ' in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])