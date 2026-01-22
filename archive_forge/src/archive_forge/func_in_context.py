from abc import ABCMeta, abstractmethod
from mlflow.utils.annotations import developer_stable
@abstractmethod
def in_context(self):
    """Determine if MLflow is running in this context.

        Returns:
            bool indicating if in this context.

        """
    pass