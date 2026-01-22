from typing import List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.output_parsers.tools import (
def parse_ai_message_to_openai_tool_action(message: BaseMessage) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    tool_actions = parse_ai_message_to_tool_action(message)
    if isinstance(tool_actions, AgentFinish):
        return tool_actions
    final_actions: List[AgentAction] = []
    for action in tool_actions:
        if isinstance(action, ToolAgentAction):
            final_actions.append(OpenAIToolAgentAction(tool=action.tool, tool_input=action.tool_input, log=action.log, message_log=action.message_log, tool_call_id=action.tool_call_id))
        else:
            final_actions.append(action)
    return final_actions