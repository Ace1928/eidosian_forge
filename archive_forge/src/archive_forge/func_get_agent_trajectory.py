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
@staticmethod
def get_agent_trajectory(steps: Union[str, Sequence[Tuple[AgentAction, str]]]) -> str:
    """Get the agent trajectory as a formatted string.

        Args:
            steps (Union[str, List[Tuple[AgentAction, str]]]): The agent trajectory.

        Returns:
            str: The formatted agent trajectory.
        """
    if isinstance(steps, str):
        return steps
    return '\n\n'.join([f'Step {i}:\nTool used: {action.tool}\nTool input: {action.tool_input}\nTool output: {output}' for i, (action, output) in enumerate(steps, 1)])