from __future__ import annotations
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.pydantic_v1 import Field
from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.generate_prompt import PROMPT
from langchain.output_parsers.regex import RegexParser
Load QA Generate Chain from LLM.