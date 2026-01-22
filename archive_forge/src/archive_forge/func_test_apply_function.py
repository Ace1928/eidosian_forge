import unittest
import torch
import pandas as pd
import concurrent.futures
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Dict, Any, Optional
def test_apply_function(self):
    activation_manager = IndegoActivation()
    input = torch.tensor(0.5)
    output = activation_manager.apply_function('relu', input)
    self.assertEqual(output, 0.5)
    output = activation_manager.apply_function('sigmoid', input)
    self.assertEqual(output, 0.5)
    output = activation_manager.apply_function('tanh', input)
    self.assertEqual(output, 0.46211716)
    output = activation_manager.apply_function('leaky_relu', input)
    self.assertEqual(output, 0.5)
    output = activation_manager.apply_function('elu', input)
    self.assertEqual(output, 0.5)
    output = activation_manager.apply_function('gelu', input)
    self.assertEqual(output, 0.5)