import copy
import random
from typing import Any, Dict, List, Optional
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, DialogPartnerWorld, validate
from parlai.core.message import Message
def get_openers(self, episode_num: int) -> Optional[List[str]]:
    """
        Override to return one or more opening messages with which to seed the self chat
        episode.

        The return value should be an array of strings, each string being a message in
        response to the string before it.
        """
    if self._openers:
        return [random.choice(self._openers)]
    return None