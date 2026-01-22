import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
def load_messages_from_context(self, context_name: str) -> List:
    """Load a lanchain prompt from a Kinetica context.

        A Kinetica Context is an object created with the Kinetica Workbench UI or with
        SQL syntax. This function will convert the data in the context to a list of
        messages that can be used as a prompt. The messages will contain a
        ``SystemMessage`` followed by pairs of ``HumanMessage``/``AIMessage`` that
        contain the samples.

        Args:
            context_name: The name of an LLM context in the database.

        Returns:
            A list of messages containing the information from the context.
        """
    sql = f"GENERATE PROMPT WITH OPTIONS (CONTEXT_NAMES = '{context_name}')"
    result = self._execute_sql(sql)
    prompt = result['Prompt']
    prompt_json = json.loads(prompt)
    request = _KdtoSuggestRequest.parse_obj(prompt_json)
    payload = request.payload
    dict_messages = []
    dict_messages.append(dict(role='system', content=payload.get_system_str()))
    dict_messages.extend(payload.get_messages())
    messages = [self._convert_message_from_dict(m) for m in dict_messages]
    return messages