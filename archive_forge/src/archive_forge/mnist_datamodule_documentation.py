import logging
import os
import random
import time
import urllib
from typing import Any, Callable, Optional, Sized, Tuple, Union
from urllib.error import HTTPError
from warnings import warn
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning_fabric.utilities.imports import _IS_WINDOWS
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
MNIST test set uses the test split.