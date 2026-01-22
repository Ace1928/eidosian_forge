import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import root_validator
@root_validator(pre=True)
def validate_layerup_sdk(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from layerup_security import LayerupSecurity as LayerupSecuritySDK
        values['client'] = LayerupSecuritySDK(api_key=values['layerup_api_key'], base_url=values['layerup_api_base_url'])
    except ImportError:
        raise ImportError('Could not import LayerupSecurity SDK. Please install it with `pip install LayerupSecurity`.')
    return values