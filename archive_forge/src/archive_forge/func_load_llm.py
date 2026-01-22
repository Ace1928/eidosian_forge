import json
from pathlib import Path
from typing import Any, Union
import yaml
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms import get_type_to_cls_dict
def load_llm(file: Union[str, Path], **kwargs: Any) -> BaseLLM:
    """Load LLM from a file."""
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    if file_path.suffix == '.json':
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix.endswith(('.yaml', '.yml')):
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError('File type must be json or yaml')
    return load_llm_from_config(config, **kwargs)