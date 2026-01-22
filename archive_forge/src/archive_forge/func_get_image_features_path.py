from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
def get_image_features_path(self, task, image_model_name, dt):
    """
        Return path dummy image features.
        """
    return self.image_features_path