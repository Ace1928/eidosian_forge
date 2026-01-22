from __future__ import annotations
from typing import Any, List, Optional, Sequence
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.agent_toolkits.nla.tool import NLATool
from langchain_community.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain_community.tools.plugin import AIPlugin
from langchain_community.utilities.requests import Requests
@classmethod
def from_llm_and_ai_plugin_url(cls, llm: BaseLanguageModel, ai_plugin_url: str, requests: Optional[Requests]=None, verbose: bool=False, **kwargs: Any) -> NLAToolkit:
    """Instantiate the toolkit from an OpenAPI Spec URL"""
    plugin = AIPlugin.from_url(ai_plugin_url)
    return cls.from_llm_and_ai_plugin(llm=llm, ai_plugin=plugin, requests=requests, verbose=verbose, **kwargs)