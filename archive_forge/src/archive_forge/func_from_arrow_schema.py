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
@classmethod
def from_arrow_schema(cls, pa_schema: pa.Schema) -> 'Features':
    """
        Construct [`Features`] from Arrow Schema.
        It also checks the schema metadata for Hugging Face Datasets features.
        Non-nullable fields are not supported and set to nullable.

        Args:
            pa_schema (`pyarrow.Schema`):
                Arrow Schema.

        Returns:
            [`Features`]
        """
    metadata_features = Features()
    if pa_schema.metadata is not None and 'huggingface'.encode('utf-8') in pa_schema.metadata:
        metadata = json.loads(pa_schema.metadata['huggingface'.encode('utf-8')].decode())
        if 'info' in metadata and 'features' in metadata['info'] and (metadata['info']['features'] is not None):
            metadata_features = Features.from_dict(metadata['info']['features'])
    metadata_features_schema = metadata_features.arrow_schema
    obj = {field.name: metadata_features[field.name] if field.name in metadata_features and metadata_features_schema.field(field.name) == field else generate_from_arrow_type(field.type) for field in pa_schema}
    return cls(**obj)