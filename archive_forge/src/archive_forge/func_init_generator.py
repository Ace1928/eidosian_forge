import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
import requests
from outlines import generate, models
def init_generator(self):
    """Load the model and initialize the generator."""
    model = models.transformers(self.model_name)
    self.generator = generate.json(model, self.schema)