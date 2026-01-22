from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
def validate_model_training_status(self) -> None:
    if self.model_training_status != 'training_complete':
        raise Exception(f'Model {self.model_id} is not ready. Please wait for training to complete.')