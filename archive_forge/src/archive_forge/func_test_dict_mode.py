import numpy as np
import pytest
from unittest.mock import Mock, patch
def test_dict_mode(self):
    config = np.show_config(mode='dicts')
    assert isinstance(config, dict)
    assert all([key in config for key in self.REQUIRED_CONFIG_KEYS]), 'Required key missing, see index of `False` with `REQUIRED_CONFIG_KEYS`'