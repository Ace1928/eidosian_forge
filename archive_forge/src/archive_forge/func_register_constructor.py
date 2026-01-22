import inspect
import warnings
from contextlib import suppress
from typing import Dict, Optional
import entrypoints
import mlflow.data
from mlflow.data.dataset import Dataset
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def register_constructor(self, constructor_fn: callable, constructor_name: Optional[str]=None) -> str:
    """Registers a dataset constructor.

        Args:
            constructor_fn: A function that accepts at least the following
                inputs and returns an instance of a subclass of
                :py:class:`mlflow.data.dataset.Dataset`:

                - name: Optional. A string dataset name
                - digest: Optional. A string dataset digest.

            constructor_name: The name of the constructor, e.g.
                "from_spark". The name must begin with the
                string "from_" or "load_". If unspecified, the `__name__`
                attribute of the `constructor_fn` is used instead and must
                begin with the string "from_" or "load_".

        Returns:
            The name of the registered constructor, e.g. "from_pandas" or "load_delta".
        """
    if constructor_name is None:
        constructor_name = constructor_fn.__name__
    DatasetRegistry._validate_constructor(constructor_fn, constructor_name)
    self.constructors[constructor_name] = constructor_fn
    return constructor_name