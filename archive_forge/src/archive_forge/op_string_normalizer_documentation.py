import locale as pylocale
import unicodedata
import warnings
import numpy as np
from onnx.reference.op_run import OpRun, RuntimeTypeError
Transforms accentuated unicode symbols into their simple counterpart.
        Source: `sklearn/feature_extraction/text.py
        <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/
        feature_extraction/text.py#L115>`_.

        Args:
            s: string The string to strip

        Returns:
            the cleaned string
        