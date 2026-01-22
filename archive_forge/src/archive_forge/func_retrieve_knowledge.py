import json
import os
from random import choice
from typing import Dict
from parlai.core.agents import create_agent, add_datapath_and_model_args
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.tasks.wizard_of_wikipedia.build import build
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import KnowledgeRetrieverAgent
from openchat.base import ParlaiGenerationAgent
def retrieve_knowledge(self, text):
    message = Message({'id': 'local_human', 'text': self.chosen_topic + self.suffix + text, 'chosen_topic': self.chosen_topic, 'episode_done': False, 'label_candidates': None})
    self.knowledge_retriever.observe(message)
    knowledge = self.knowledge_retriever.act()['checked_sentence']
    knowledge = self.TOKEN_KNOWLEDGE + knowledge + self.TOKEN_END_KNOWLEDGE
    return knowledge + self.suffix + text