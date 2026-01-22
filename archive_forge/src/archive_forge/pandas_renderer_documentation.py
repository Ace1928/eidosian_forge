import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator

    Rendering the data statistics in a HTML format.

    Args:
        inputs: Either a single "glimpse" DataFrame that contains the statistics, or a
            collection of (name, DataFrame) pairs where each pair names a separate "glimpse"
            and they are all visualized in comparison mode.

    Returns:
        None
    