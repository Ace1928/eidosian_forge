import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def get_model_files(model_type: str, frameworks: Optional[List[str]]=None) -> Dict[str, Union[Path, List[Path]]]:
    """
    Retrieves all the files associated to a model.

    Args:
        model_type (`str`): A valid model type (like "bert" or "gpt2")
        frameworks (`List[str]`, *optional*):
            If passed, will only keep the model files corresponding to the passed frameworks.

    Returns:
        `Dict[str, Union[Path, List[Path]]]`: A dictionary with the following keys:
        - **doc_file** -- The documentation file for the model.
        - **model_files** -- All the files in the model module.
        - **test_files** -- The test files for the model.
    """
    module_name = model_type_to_module_name(model_type)
    model_module = TRANSFORMERS_PATH / 'models' / module_name
    model_files = list(model_module.glob('*.py'))
    model_files = filter_framework_files(model_files, frameworks=frameworks)
    doc_file = REPO_PATH / 'docs' / 'source' / 'en' / 'model_doc' / f'{model_type}.md'
    test_files = [f'test_modeling_{module_name}.py', f'test_modeling_tf_{module_name}.py', f'test_modeling_flax_{module_name}.py', f'test_tokenization_{module_name}.py', f'test_image_processing_{module_name}.py', f'test_feature_extraction_{module_name}.py', f'test_processor_{module_name}.py']
    test_files = filter_framework_files(test_files, frameworks=frameworks)
    test_files = [REPO_PATH / 'tests' / 'models' / module_name / f for f in test_files]
    test_files = [f for f in test_files if f.exists()]
    return {'doc_file': doc_file, 'model_files': model_files, 'module_name': module_name, 'test_files': test_files}