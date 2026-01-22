import json
from json import JSONDecodeError
from typing import List, Union
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import MultiActionAgentOutputParser
def parse_ai_message_to_tool_action(message: BaseMessage) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f'Expected an AI message got {type(message)}')
    actions: List = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get('tool_calls'):
            return AgentFinish(return_values={'output': message.content}, log=str(message.content))
        tool_calls = []
        for tool_call in message.additional_kwargs['tool_calls']:
            function = tool_call['function']
            function_name = function['name']
            try:
                args = json.loads(function['arguments'] or '{}')
                tool_calls.append(ToolCall(name=function_name, args=args, id=tool_call['id']))
            except JSONDecodeError:
                raise OutputParserException(f'Could not parse tool input: {function} because the `arguments` is not valid JSON.')
    for tool_call in tool_calls:
        function_name = tool_call['name']
        _tool_input = tool_call['args']
        if '__arg1' in _tool_input:
            tool_input = _tool_input['__arg1']
        else:
            tool_input = _tool_input
        content_msg = f'responded: {message.content}\n' if message.content else '\n'
        log = f'\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n'
        actions.append(ToolAgentAction(tool=function_name, tool_input=tool_input, log=log, message_log=[message], tool_call_id=tool_call['id']))
    return actions