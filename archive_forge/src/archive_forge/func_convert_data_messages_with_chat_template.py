from __future__ import annotations
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types.llm import (
def convert_data_messages_with_chat_template(data, tokenizer):
    """For the Chat inference task, apply chat template to messages to create prompt."""
    if 'messages' in data.columns:
        messages = data.pop('messages').tolist()[0]
    else:
        raise MlflowException("The 'messages' field is required for the Chat inference task.")
    try:
        messages_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        raise MlflowException(f'Failed to apply chat template: {e}')
    data['prompt'] = messages_str