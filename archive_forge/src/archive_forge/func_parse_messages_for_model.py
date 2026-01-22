from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field
from requests import Response
from requests.exceptions import HTTPError
def parse_messages_for_model(self, messages: List[BaseMessage]) -> List[Dict[str, Union[str, List[Union[str, Dict[Any, Any]]]]]]:
    """
        Parses messages from LangChain's format to the format expected by
        the MariTalk API.

        Parameters:
            messages (List[BaseMessage]): A list of messages in LangChain
            format to be parsed.

        Returns:
            A list of messages formatted for the MariTalk API.
        """
    parsed_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = 'user'
        elif isinstance(message, AIMessage):
            role = 'assistant'
        elif isinstance(message, SystemMessage):
            role = 'system'
        parsed_messages.append({'role': role, 'content': message.content})
    return parsed_messages