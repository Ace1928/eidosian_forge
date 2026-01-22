import unittest
import torch
from trl.core import masked_mean, masked_var, masked_whiten, whiten
def test_masked_var(self):
    assert torch.var(self.test_input_unmasked) == masked_var(self.test_input, self.test_mask)