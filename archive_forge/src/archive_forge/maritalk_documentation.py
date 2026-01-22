from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import Field
from requests import Response
from requests.exceptions import HTTPError

        Identifies the key parameters of the chat model for logging
        or tracking purposes.

        Returns:
            A dictionary of the key configuration parameters.
        