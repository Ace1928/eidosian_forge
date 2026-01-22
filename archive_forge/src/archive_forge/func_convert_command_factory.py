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
def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(args.model_name, args.local_dir, args.max_error, args.new_weights, args.no_pr, args.push, args.extra_commit_description, args.override_model_class)