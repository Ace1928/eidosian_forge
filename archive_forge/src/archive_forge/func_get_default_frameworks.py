import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def get_default_frameworks():
    """
    Returns the list of frameworks (PyTorch, TensorFlow, Flax) that are installed in the environment.
    """
    frameworks = []
    if is_torch_available():
        frameworks.append('pt')
    if is_tf_available():
        frameworks.append('tf')
    if is_flax_available():
        frameworks.append('flax')
    return frameworks