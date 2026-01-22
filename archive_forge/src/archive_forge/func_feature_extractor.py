import re
import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin
@property
def feature_extractor(self):
    warnings.warn('`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.', FutureWarning)
    return self.image_processor