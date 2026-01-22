import json
import logging
from datetime import datetime
from typing import List, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
def reload_cache(self) -> None:
    """reloads messages from database to cache"""
    self.cache.clear()
    self._load_messages_to_cache()