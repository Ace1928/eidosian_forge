import unittest
import logging
from openchat.agents.blender import BlenderGenerationAgent
from openchat.agents.dialogpt import DialoGPTAgent
from openchat.agents.dodecathlon import DodecathlonAgent
from openchat.agents.reddit import RedditAgent
from openchat.agents.offensive import SafetyAgent
from openchat.agents.unlikelihood import UnlikelihoodAgent
from openchat.agents.wow import WizardOfWikipediaGenerationAgent
from openchat.base import WizardOfWikipediaAgent
def test_blender(self):
    for model in BlenderGenerationAgent.available_models():
        self.model_unittest(model, BlenderGenerationAgent)