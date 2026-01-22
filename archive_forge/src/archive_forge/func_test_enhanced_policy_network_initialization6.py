import unittest  # Importing the unittest module for creating and running tests
import torch  # Importing the PyTorch library for tensor computations and neural network operations
import sys  # Importing the sys module to interact with the Python runtime environment
import asyncio  # Importing the asyncio module for writing single-threaded concurrent code using coroutines
import logging  # Importing the logging module to enable logging of messages of various severity levels
from unittest.mock import (
import numpy as np  # Importing the numpy library for numerical operations on arrays
import unittest
import numpy as np
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
from ActivationDictionary import (
from IndegoAdaptAct import (
from IndegoLogging import configure_logging
def test_enhanced_policy_network_initialization6(self):
    in_dim = 15
    out_dim = 12
    layers = [30, 45, 60, 75, 90]
    hyperparams = {'learning_rate': 0.001, 'regularization_factor': 0.0001, 'discount_factor': 0.97}
    network = EnhancedPolicyNetwork(in_dim, out_dim, layers, hyperparams)
    self.assertEqual(network.input_dim, in_dim)
    self.assertEqual(network.output_dim, out_dim)
    self.assertEqual(network.layer_info, layers)
    self.assertEqual(network.hyperparameters, hyperparams)
    self.assertEqual(len(network.model), 2 * len(layers) + 1)