from the model zoo.
from parlai.core.agents import Agent, create_agent, create_agent_from_shared
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
from parlai.zoo.wizard_of_wikipedia.knowledge_retriever import download
import json
import os
def maintain_retriever_history(self, obs, actor_id='apprentice'):
    """
        Maintain texts retrieved by the retriever to mimic the set-up from the data
        collection for the task.
        """
    if actor_id == 'apprentice':
        if self.retriever_history['episode_done']:
            self.retriever_history['chosen_topic'] = ''
            self.retriever_history['wizard'] = ''
            self.retriever_history['apprentice'] = ''
            self.retriever_history['episode_done'] = False
            self.dialogue_history = []
        if 'chosen_topic' in obs:
            self.retriever_history['chosen_topic'] = obs['chosen_topic']
        if 'text' in obs:
            self.retriever_history['apprentice'] = obs['text']
        self.retriever_history['episode_done'] = obs['episode_done']
    elif not self.retriever_history['episode_done']:
        self.retriever_history['wizard'] = obs['text']
    self.dialogue_history.append({'id': actor_id, 'text': obs['text']})