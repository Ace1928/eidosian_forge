import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin

        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        