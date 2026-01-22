import json
import numpy as np
import os
from ray.tune import Trainable
Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    