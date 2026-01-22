from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
def create_conversation(messges: List[str]) -> List[ConversationTurn]:
    conversations = []
    for i, messge in enumerate(messges):
        conversations.append(ConversationTurn(message=messge, agent_type=AgentType.USER if i % 2 == 0 else AgentType.AGENT))
    return conversations