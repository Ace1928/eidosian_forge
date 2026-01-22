import collections
import inspect
import os
import pickle
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import base_tuner
Performs hyperparameter search.

        Args:
            X: See docstring for `model.fit` for the `sklearn` Models being
                tuned.
            y: See docstring for `model.fit` for the `sklearn` Models being
                tuned.
            sample_weight: Optional. See docstring for `model.fit` for the
                `sklearn` Models being tuned.
            groups: Optional. Required for `sklearn.model_selection` Splitter
                classes that split based on group labels (For example, see
                `sklearn.model_selection.GroupKFold`).
        