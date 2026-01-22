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
def model_unittest(self, model_name, model_class):
    model = model_class(model=model_name, device='cpu')
    if isinstance(model, WizardOfWikipediaAgent):
        model.set_topic('Guitar')
    output = model.predict('hello.')
    logging.info(f'{model} testing is success: {output}')
    self.assertIsInstance(output, dict)