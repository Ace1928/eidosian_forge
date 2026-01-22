import json
import logging
from mlflow.protos.databricks_pb2 import (
@classmethod
def invalid_parameter_value(cls, message, **kwargs):
    """Constructs an `MlflowException` object with the `INVALID_PARAMETER_VALUE` error code.

        Args:
            message: The message describing the error that occurred. This will be included in the
                exception's serialized JSON representation.
            kwargs: Additional key-value pairs to include in the serialized JSON representation
                of the MlflowException.
        """
    return cls(message, error_code=INVALID_PARAMETER_VALUE, **kwargs)