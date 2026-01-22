import json
import logging
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.slack.base import SlackBaseTool
Tool that gets Slack messages.