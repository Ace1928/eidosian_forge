import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import numpy as np
import asyncio
from typing import Dict, Any, List, cast, Tuple, Callable, Optional, Union
import sys
import torch.optim as optim
import os
from ActivationDictionary import ActivationDictionary
from IndegoLogging import configure_logging
def log_decision(layer_output: torch.Tensor, chosen_activation: str, reward: float) -> None:
    try:
        layer_output_list: List[float] = layer_output.tolist()
        logger.debug(f'Converted layer output to list: {layer_output_list}')
        log_message: str = f'Decision Log - Layer Output: {layer_output_list}, Chosen Activation Function: {chosen_activation}, Reward Received: {reward}'
        logger.debug(f'Constructed log message: {log_message}')
        logger.info(log_message)
        logger.debug('Successfully logged the decision details at the INFO level.')
    except Exception as e:
        error_message: str = f'An error occurred while logging the decision: {e}'
        logger.error(error_message, exc_info=True)
        raise RuntimeError(f'Logging of the decision failed due to: {e}') from e