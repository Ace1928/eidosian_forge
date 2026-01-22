from typing import List, Optional, Union
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.utils import get_from_env
Clear session memory from Neo4j