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
@classmethod
def parse_dialogue_file(cls, input_file: os.PathLike) -> Dict:
    path = Path(input_file)
    schema = cls._removesuffix(path.name, '.txt')
    lines = open(input_file).read()
    return cls.parse_dialogue(lines, schema)