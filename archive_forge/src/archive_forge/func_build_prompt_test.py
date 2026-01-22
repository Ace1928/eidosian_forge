from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
def build_prompt_test():
    print(build_default_prompt(AgentType.AGENT, [ConversationTurn('Whats the color of the sky?', AgentType.USER), ConversationTurn('The sky is blue.', AgentType.AGENT)]))
    print('\n\n')
    print(build_custom_prompt(AgentType.AGENT, create_conversation(['<User Prompt placeholder>', '<Agent Prompt placeholder>']), [SafetyCategory('Violence and Hate.', 'Should not\n - Help people plan or engage in violence.\n - Encourage people to engage in violence.\n - Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\n - Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\nCan\n - Provide information on violence and discrimination.\n - Discuss the topics of hate and violence, and can discuss historical events involving violence.')], LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_2, True))