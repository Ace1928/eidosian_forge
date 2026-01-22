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
def encode_nested_example(schema, obj, level=0):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be encoded.
    If the first element needs to be encoded, then all the elements of the list will be encoded, otherwise they'll stay the same.
    """
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError('Got None but expected a dictionary instead')
        return {k: encode_nested_example(schema[k], obj.get(k), level=level + 1) for k in schema} if obj is not None else None
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if encode_nested_example(sub_schema, first_elmt, level=level + 1) != first_elmt:
                    return [encode_nested_example(sub_schema, o, level=level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        if isinstance(schema.feature, dict):
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                for k in schema.feature:
                    list_dict[k] = [encode_nested_example(schema.feature[k], o.get(k), level=level + 1) for o in obj]
                return list_dict
            else:
                for k in schema.feature:
                    list_dict[k] = [encode_nested_example(schema.feature[k], o, level=level + 1) for o in obj[k]] if k in obj else None
                return list_dict
        if isinstance(obj, str):
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, schema.feature):
                        break
                if not isinstance(first_elmt, list) or encode_nested_example(schema.feature, first_elmt, level=level + 1) != first_elmt:
                    return [encode_nested_example(schema.feature, o, level=level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, (Audio, Image, ClassLabel, TranslationVariableLanguages, Value, _ArrayXD)):
        return schema.encode_example(obj) if obj is not None else None
    return obj