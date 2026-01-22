import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
def require_decoding(feature: FeatureType, ignore_decode_attribute: bool=False) -> bool:
    """Check if a (possibly nested) feature requires decoding.

    Args:
        feature (FeatureType): the feature type to be checked
        ignore_decode_attribute (:obj:`bool`, default ``False``): Whether to ignore the current value
            of the `decode` attribute of the decodable feature types.
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any((require_decoding(f) for f in feature.values()))
    elif isinstance(feature, (list, tuple)):
        return require_decoding(feature[0])
    elif isinstance(feature, Sequence):
        return require_decoding(feature.feature)
    else:
        return hasattr(feature, 'decode_example') and (feature.decode if not ignore_decode_attribute else True)