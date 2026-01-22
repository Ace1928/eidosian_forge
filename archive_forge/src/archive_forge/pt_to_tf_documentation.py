import inspect
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
import huggingface_hub
import numpy as np
from packaging import version
from .. import (
from ..utils import TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, logging
from . import BaseTransformersCLICommand

        Returns the right inputs for the model, based on its signature.
        