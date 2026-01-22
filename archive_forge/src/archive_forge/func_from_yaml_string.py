import re
import textwrap
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import yaml
from huggingface_hub import DatasetCardData
from ..config import METADATA_CONFIGS_FIELD
from ..info import DatasetInfo, DatasetInfosDict
from ..naming import _split_re
from ..utils.logging import get_logger
from .deprecation_utils import deprecated
@classmethod
def from_yaml_string(cls, string: str) -> 'DatasetMetadata':
    """Loads and validates the dataset metadata from a YAML string

        Args:
            string (:obj:`str`): The YAML string

        Returns:
            :class:`DatasetMetadata`: The dataset's metadata

        Raises:
            :obj:`TypeError`: If the dataset's metadata is invalid
        """
    metadata_dict = yaml.load(string, Loader=_NoDuplicateSafeLoader) or {}
    metadata_dict = {key.replace('-', '_') if key.replace('-', '_') in cls._FIELDS_WITH_DASHES else key: value for key, value in metadata_dict.items()}
    return cls(**metadata_dict)