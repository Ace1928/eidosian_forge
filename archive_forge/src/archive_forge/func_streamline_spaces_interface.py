from __future__ import annotations
import base64
import math
import re
import warnings
import httpx
import yaml
from huggingface_hub import InferenceClient
from gradio import components
def streamline_spaces_interface(config: dict) -> dict:
    """Streamlines the interface config dictionary to remove unnecessary keys."""
    config['inputs'] = [components.get_component_instance(component) for component in config['input_components']]
    config['outputs'] = [components.get_component_instance(component) for component in config['output_components']]
    parameters = {'article', 'description', 'flagging_options', 'inputs', 'outputs', 'title'}
    config = {k: config[k] for k in parameters}
    return config