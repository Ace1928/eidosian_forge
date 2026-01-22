import warnings
from ...utils.import_utils import check_if_transformers_greater
from .decoder_models import (
from .encoder_models import (
@staticmethod
def requires_strict_validation(model_type: str) -> bool:
    """
        Returns True if the architecture requires to make sure all conditions of `validate_bettertransformer` are met.

        Args:
            model_type (`str`):
                The model type to check.
        """
    return model_type not in BetterTransformerManager.NOT_REQUIRES_STRICT_VALIDATION