import logging
from typing import TYPE_CHECKING, Dict, Optional, Union
import numpy as np
import torch
from ray.air._internal.torch_utils import convert_ndarray_batch_to_torch_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.torch import TorchCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
Run inference on data batch.

        If the provided data is a single array or a dataframe/table with a single
        column, it will be converted into a single PyTorch tensor before being
        inputted to the model.

        If the provided data is a multi-column table or a dict of numpy arrays,
        it will be converted into a dict of tensors before being inputted to the
        model. This is useful for multi-modal inputs (for example your model accepts
        both image and text).

        Args:
            data: A batch of input data of ``DataBatchType``.
            dtype: The dtypes to use for the tensors. Either a single dtype for all
                tensors or a mapping from column name to dtype.

        Returns:
            DataBatchType: Prediction result. The return type will be the same as the
                input type.

        Example:

            .. testcode::

                    import numpy as np
                    import pandas as pd
                    import torch
                    import ray
                    from ray.train.torch import TorchPredictor

                    # Define a custom PyTorch module
                    class CustomModule(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.linear1 = torch.nn.Linear(1, 1)
                            self.linear2 = torch.nn.Linear(1, 1)

                        def forward(self, input_dict: dict):
                            out1 = self.linear1(input_dict["A"].unsqueeze(1))
                            out2 = self.linear2(input_dict["B"].unsqueeze(1))
                            return out1 + out2

                    # Set manul seed so we get consistent output
                    torch.manual_seed(42)

                    # Use Standard PyTorch model
                    model = torch.nn.Linear(2, 1)
                    predictor = TorchPredictor(model=model)
                    # Define our data
                    data = np.array([[1, 2], [3, 4]])
                    predictions = predictor.predict(data, dtype=torch.float)
                    print(f"Standard model predictions: {predictions}")
                    print("---")

                    # Use Custom PyTorch model with TorchPredictor
                    predictor = TorchPredictor(model=CustomModule())
                    # Define our data and predict Customer model with TorchPredictor
                    data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
                    predictions = predictor.predict(data, dtype=torch.float)
                    print(f"Custom model predictions: {predictions}")

            .. testoutput::

                Standard model predictions: {'predictions': array([[1.5487633],
                       [3.8037925]], dtype=float32)}
                ---
                Custom model predictions:     predictions
                0  [0.61623406]
                1    [2.857038]
        