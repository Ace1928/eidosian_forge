import json
from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_community.tools.connery.models import Action
from langchain_community.tools.connery.tool import ConneryAction
def list_actions(self) -> List[ConneryAction]:
    """
        Returns the list of actions available in the Connery Runner.
        Returns:
            List[ConneryAction]: The list of actions available in the Connery Runner.
        """
    return [ConneryAction.create_instance(action, self) for action in self._list_actions()]