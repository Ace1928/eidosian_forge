import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, Union
import numpy as np
import tensorflow as tf
from ray.air._internal.tensorflow_utils import convert_ndarray_batch_to_tf_tensor_batch
from ray.train._internal.dl_predictor import DLPredictor
from ray.train.predictor import DataBatchType
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util import log_once
from ray.util.annotations import DeveloperAPI, PublicAPI
Run inference on data batch.

        If the provided data is a single array or a dataframe/table with a single
        column, it will be converted into a single Tensorflow tensor before being
        inputted to the model.

        If the provided data is a multi-column table or a dict of numpy arrays,
        it will be converted into a dict of tensors before being inputted to the
        model. This is useful for multi-modal inputs (for example your model accepts
        both image and text).

        Args:
            data: A batch of input data. Either a pandas DataFrame or numpy
                array.
            dtype: The dtypes to use for the tensors. Either a single dtype for all
                tensors or a mapping from column name to dtype.

        Examples:

        .. testcode::

            import numpy as np
            import tensorflow as tf
            from ray.train.tensorflow import TensorflowPredictor

            def build_model():
                return tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=()),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(1),
                    ]
                )

            weights = [np.array([[2.0]]), np.array([0.0])]
            predictor = TensorflowPredictor(model=build_model())

            data = np.asarray([1, 2, 3])
            predictions = predictor.predict(data)

            import pandas as pd
            import tensorflow as tf
            from ray.train.tensorflow import TensorflowPredictor

            def build_model():
                input1 = tf.keras.layers.Input(shape=(1,), name="A")
                input2 = tf.keras.layers.Input(shape=(1,), name="B")
                merged = tf.keras.layers.Concatenate(axis=1)([input1, input2])
                output = tf.keras.layers.Dense(2, input_dim=2)(merged)
                return tf.keras.models.Model(
                    inputs=[input1, input2], outputs=output)

            predictor = TensorflowPredictor(model=build_model())

            # Pandas dataframe.
            data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])

            predictions = predictor.predict(data)

        Returns:
            DataBatchType: Prediction result. The return type will be the same as the
                input type.
        