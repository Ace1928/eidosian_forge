from __future__ import annotations
import json
import logging
import re
from typing import Optional, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain.agents.agent import AgentOutputParser
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers import OutputFixingParser
@classmethod
def from_llm(cls, llm: Optional[BaseLanguageModel]=None, base_parser: Optional[StructuredChatOutputParser]=None) -> StructuredChatOutputParserWithRetries:
    if llm is not None:
        base_parser = base_parser or StructuredChatOutputParser()
        output_fixing_parser: OutputFixingParser = OutputFixingParser.from_llm(llm=llm, parser=base_parser)
        return cls(output_fixing_parser=output_fixing_parser)
    elif base_parser is not None:
        return cls(base_parser=base_parser)
    else:
        return cls()