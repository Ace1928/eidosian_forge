import re
from typing import (
from langchain_core.agents import AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.tools import BaseTool
from langchain.callbacks.manager import (
from langchain.chains.llm import LLMChain
from langchain.evaluation.agents.trajectory_eval_prompt import (
from langchain.evaluation.schema import AgentTrajectoryEvaluator, LLMEvalChain
@property
def output_keys(self) -> List[str]:
    """Get the output keys for the chain.

        Returns:
            List[str]: The output keys.
        """
    return ['score', 'reasoning']