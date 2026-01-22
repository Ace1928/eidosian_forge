from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
def build_custom_prompt(agent_type: AgentType, conversations: List[ConversationTurn], categories: List[SafetyCategory], category_short_name_prefix: str, prompt_template: str, with_policy: bool=False):
    categories_str = '\n'.join([f'{category_short_name_prefix}{i + 1}: {c.name}' + (f'\n{c.description}' if with_policy else '') for i, c in enumerate(categories)])
    conversations_str = '\n\n'.join([f'{t.agent_type.value}: {t.message}' for t in conversations])
    return prompt_template.substitute(agent_type=agent_type.value, categories=categories_str, conversations=conversations_str)