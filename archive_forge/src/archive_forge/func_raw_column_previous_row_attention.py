import os
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
def raw_column_previous_row_attention(layer):
    return _RawColumnPreviousRowAttention[layer % 3]