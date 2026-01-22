from __future__ import annotations
import re
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain.callbacks.manager import Callbacks
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.llm import LLMChain
from langchain.evaluation.criteria.prompt import PROMPT, PROMPT_WITH_REFERENCES
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.schema import RUN_KEY
@classmethod
def resolve_criteria(cls, criteria: Optional[Union[CRITERIA_TYPE, str]]) -> Dict[str, str]:
    """Resolve the criteria to evaluate.

        Parameters
        ----------
        criteria : CRITERIA_TYPE
            The criteria to evaluate the runs against. It can be:
                -  a mapping of a criterion name to its description
                -  a single criterion name present in one of the default criteria
                -  a single `ConstitutionalPrinciple` instance

        Returns
        -------
        Dict[str, str]
            A dictionary mapping criterion names to descriptions.

        Examples
        --------
        >>> criterion = "relevance"
        >>> CriteriaEvalChain.resolve_criteria(criteria)
        {'relevance': 'Is the submission referring to a real quote from the text?'}
        """
    return resolve_criteria(criteria)