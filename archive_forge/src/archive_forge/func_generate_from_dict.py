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
def generate_from_dict(obj: Any):
    """Regenerate the nested feature object from a deserialized dict.
    We use the '_type' fields to get the dataclass name to load.

    generate_from_dict is the recursive helper for Features.from_dict, and allows for a convenient constructor syntax
    to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
    a :class:`DatasetInfo` that was dumped to a JSON object. This acts as an analogue to
    :meth:`Features.from_arrow_schema` and handles the recursive field-by-field instantiation, but doesn't require any
    mapping to/from pyarrow, except for the fact that it takes advantage of the mapping of pyarrow primitive dtypes
    that :class:`Value` automatically performs.
    """
    if isinstance(obj, list):
        return [generate_from_dict(value) for value in obj]
    if '_type' not in obj or isinstance(obj['_type'], dict):
        return {key: generate_from_dict(value) for key, value in obj.items()}
    obj = dict(obj)
    class_type = globals()[obj.pop('_type')]
    if class_type == Sequence:
        return Sequence(feature=generate_from_dict(obj['feature']), length=obj.get('length', -1))
    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})