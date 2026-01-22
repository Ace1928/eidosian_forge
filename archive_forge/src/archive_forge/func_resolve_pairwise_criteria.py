from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Union
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.comparison.prompt import (
from langchain.evaluation.criteria.eval_chain import (
from langchain.evaluation.schema import LLMEvalChain, PairwiseStringEvaluator
from langchain.schema import RUN_KEY
def resolve_pairwise_criteria(criteria: Optional[Union[CRITERIA_TYPE, str, List[CRITERIA_TYPE]]]) -> dict:
    """Resolve the criteria for the pairwise evaluator.

    Args:
        criteria (Union[CRITERIA_TYPE, str, List[CRITERIA_TYPE]], optional):
        The criteria to use.

    Returns:
        dict: The resolved criteria.

    """
    if criteria is None:
        _default_criteria = [Criteria.HELPFULNESS, Criteria.RELEVANCE, Criteria.CORRECTNESS, Criteria.DEPTH]
        return {k.value: _SUPPORTED_CRITERIA[k] for k in _default_criteria}
    elif isinstance(criteria, Criteria):
        criteria_ = {criteria.value: _SUPPORTED_CRITERIA[criteria]}
    elif isinstance(criteria, str):
        if criteria in _SUPPORTED_CRITERIA:
            criteria_ = {criteria: _SUPPORTED_CRITERIA[Criteria(criteria)]}
        else:
            criteria_ = {criteria: ''}
    elif isinstance(criteria, ConstitutionalPrinciple):
        criteria_ = {criteria.name: criteria.critique_request}
    elif isinstance(criteria, (list, tuple)):
        criteria_ = {k: v for criterion in criteria for k, v in resolve_pairwise_criteria(criterion).items()}
    else:
        if not criteria:
            raise ValueError('Criteria cannot be empty. Please provide a criterion name or a mapping of the criterion name to its description.')
        criteria_ = dict(criteria)
    return criteria_