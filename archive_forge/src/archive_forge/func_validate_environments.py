from typing import Any, Dict, List, Optional, Union, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_community.utilities.arcee import ArceeWrapper, DALMFilter
@root_validator(pre=False)
def validate_environments(cls, values: Dict) -> Dict:
    """Validate Arcee environment variables."""
    values['arcee_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'arcee_api_key', 'ARCEE_API_KEY'))
    values['arcee_api_url'] = get_from_dict_or_env(values, 'arcee_api_url', 'ARCEE_API_URL')
    values['arcee_app_url'] = get_from_dict_or_env(values, 'arcee_app_url', 'ARCEE_APP_URL')
    values['arcee_api_version'] = get_from_dict_or_env(values, 'arcee_api_version', 'ARCEE_API_VERSION')
    if values.get('model_kwargs'):
        kw = values['model_kwargs']
        if kw.get('size') is not None:
            if not kw.get('size') >= 0:
                raise ValueError('`size` must be positive')
        if kw.get('filters') is not None:
            if not isinstance(kw.get('filters'), List):
                raise ValueError('`filters` must be a list')
            for f in kw.get('filters'):
                DALMFilter(**f)
    return values