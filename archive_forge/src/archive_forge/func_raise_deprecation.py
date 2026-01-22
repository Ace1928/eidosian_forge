from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.prompt import (
from langchain.chains.sequential import SequentialChain
@root_validator(pre=True)
def raise_deprecation(cls, values: Dict) -> Dict:
    if 'llm' in values:
        warnings.warn('Directly instantiating an LLMCheckerChain with an llm is deprecated. Please instantiate with question_to_checked_assertions_chain or using the from_llm class method.')
        if 'question_to_checked_assertions_chain' not in values and values['llm'] is not None:
            question_to_checked_assertions_chain = _load_question_to_checked_assertions_chain(values['llm'], values.get('create_draft_answer_prompt', CREATE_DRAFT_ANSWER_PROMPT), values.get('list_assertions_prompt', LIST_ASSERTIONS_PROMPT), values.get('check_assertions_prompt', CHECK_ASSERTIONS_PROMPT), values.get('revised_answer_prompt', REVISED_ANSWER_PROMPT))
            values['question_to_checked_assertions_chain'] = question_to_checked_assertions_chain
    return values