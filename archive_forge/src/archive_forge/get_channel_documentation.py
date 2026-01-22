import json
import logging
from typing import Any, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools.slack.base import SlackBaseTool
Tool that gets Slack channel information.