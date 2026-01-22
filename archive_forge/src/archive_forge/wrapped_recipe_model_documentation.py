from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        