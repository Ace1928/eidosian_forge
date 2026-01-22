from __future__ import annotations
import json
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains.llm import LLMChain
from langchain.chains.query_constructor.ir import (
from langchain.chains.query_constructor.parser import get_parser
from langchain.chains.query_constructor.prompt import (
from langchain.chains.query_constructor.schema import AttributeInfo
def get_query_constructor_prompt(document_contents: str, attribute_info: Sequence[Union[AttributeInfo, dict]], *, examples: Optional[Sequence]=None, allowed_comparators: Sequence[Comparator]=tuple(Comparator), allowed_operators: Sequence[Operator]=tuple(Operator), enable_limit: bool=False, schema_prompt: Optional[BasePromptTemplate]=None, **kwargs: Any) -> BasePromptTemplate:
    """Create query construction prompt.

    Args:
        document_contents: The contents of the document to be queried.
        attribute_info: A list of AttributeInfo objects describing
            the attributes of the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators.
        allowed_operators: Sequence of allowed operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        **kwargs: Additional named params to pass to FewShotPromptTemplate init.

    Returns:
        A prompt template that can be used to construct queries.
    """
    default_schema_prompt = SCHEMA_WITH_LIMIT_PROMPT if enable_limit else DEFAULT_SCHEMA_PROMPT
    schema_prompt = schema_prompt or default_schema_prompt
    attribute_str = _format_attribute_info(attribute_info)
    schema = schema_prompt.format(allowed_comparators=' | '.join(allowed_comparators), allowed_operators=' | '.join(allowed_operators))
    if examples and isinstance(examples[0], tuple):
        examples = construct_examples(examples)
        example_prompt = USER_SPECIFIED_EXAMPLE_PROMPT
        prefix = PREFIX_WITH_DATA_SOURCE.format(schema=schema, content=document_contents, attributes=attribute_str)
        suffix = SUFFIX_WITHOUT_DATA_SOURCE.format(i=len(examples) + 1)
    else:
        examples = examples or (EXAMPLES_WITH_LIMIT if enable_limit else DEFAULT_EXAMPLES)
        example_prompt = EXAMPLE_PROMPT
        prefix = DEFAULT_PREFIX.format(schema=schema)
        suffix = DEFAULT_SUFFIX.format(i=len(examples) + 1, content=document_contents, attributes=attribute_str)
    return FewShotPromptTemplate(examples=list(examples), example_prompt=example_prompt, input_variables=['query'], suffix=suffix, prefix=prefix, **kwargs)