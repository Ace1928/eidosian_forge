import warnings
from ...utils.import_utils import check_if_transformers_greater
from .decoder_models import (
from .encoder_models import (
@staticmethod
def requires_nested_tensor(model_type: str) -> bool:
    """
        Returns True if the BetterTransformer implementation for a given architecture uses nested tensors, False otherwise.

        Args:
            model_type (`str`):
                The model type to check.
        """
    return model_type not in BetterTransformerManager.NOT_REQUIRES_NESTED_TENSOR