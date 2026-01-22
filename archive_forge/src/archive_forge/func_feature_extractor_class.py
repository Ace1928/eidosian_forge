import re
import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin
@property
def feature_extractor_class(self):
    warnings.warn('`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.', FutureWarning)
    return self.image_processor_class