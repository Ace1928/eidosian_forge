from abc import ABC, abstractmethod
from dataclasses import dataclass
from openchat.base import BaseAgent, DecoderLM
def clear_histories(self, user_id):
    self.histories[user_id] = {'user_message': [], 'bot_message': [], 'model_input': '', 'prefix': [], 'chosen_topic': ''}