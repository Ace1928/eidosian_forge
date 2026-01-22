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
def from_llm_and_spec(cls, llm: BaseLanguageModel, spec: OpenAPISpec, requests: Optional[Requests]=None, verbose: bool=False, **kwargs: Any) -> NLAToolkit:
    """Instantiate the toolkit by creating tools for each operation."""
    http_operation_tools = cls._get_http_operation_tools(llm=llm, spec=spec, requests=requests, verbose=verbose, **kwargs)
    return cls(nla_tools=http_operation_tools)