from abc import ABCMeta, abstractmethod
from mlflow.utils.annotations import developer_stable
def can_build_image(self):
    """
        Returns:
            True if this flavor has a `build_image` method defined for building a docker
            container capable of serving the model, False otherwise.
        """
    return callable(getattr(self.__class__, 'build_image', None))