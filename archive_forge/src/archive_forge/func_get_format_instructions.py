import json
import re
from typing import Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.agent import AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
def get_format_instructions(self) -> str:
    """Returns formatting instructions for the given output parser."""
    return self.format_instructions