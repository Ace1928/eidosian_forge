import logging
from typing import Union
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available

    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    