from __future__ import annotations
from typing import List, Optional, Sequence
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.utils.input import get_color_mapping, print_text
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
@classmethod
def from_llms(cls, llms: List[BaseLLM], prompt: Optional[PromptTemplate]=None) -> ModelLaboratory:
    """Initialize with LLMs to experiment with and optional prompt.

        Args:
            llms: list of LLMs to experiment with
            prompt: Optional prompt to use to prompt the LLMs. Defaults to None.
                If a prompt was provided, it should only have one input variable.
        """
    if prompt is None:
        prompt = PromptTemplate(input_variables=['_input'], template='{_input}')
    chains = [LLMChain(llm=llm, prompt=prompt) for llm in llms]
    names = [str(llm) for llm in llms]
    return cls(chains, names=names)