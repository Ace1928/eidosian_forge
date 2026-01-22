import parlai.utils.logging as logging
from abc import ABC, abstractmethod
from typing import Dict
def check_agent(self, model) -> str:
    model = model.lower()
    available_models = self.available_models()
    assert model in available_models, f'param `model` must be one of {available_models}'
    return model