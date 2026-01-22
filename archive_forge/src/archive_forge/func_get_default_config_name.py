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
def get_default_config_name(self) -> Optional[str]:
    default_config_name = None
    for config_name, metadata_config in self.items():
        if len(self) == 1 or config_name == 'default' or metadata_config.get('default'):
            if default_config_name is None:
                default_config_name = config_name
            else:
                raise ValueError(f"Dataset has several default configs: '{default_config_name}' and '{config_name}'.")
    return default_config_name