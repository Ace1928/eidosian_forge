import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, get_file_from_repo, logging
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
def feature_extractor_class_from_name(class_name: str):
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)
            module = importlib.import_module(f'.{module_name}', 'transformers.models')
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue
    for _, extractor in FEATURE_EXTRACTOR_MAPPING._extra_content.items():
        if getattr(extractor, '__name__', None) == class_name:
            return extractor
    main_module = importlib.import_module('transformers')
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)
    return None