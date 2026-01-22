import warnings
import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.utils.import_hooks import register_post_import_hook
def get_evaluator(self, evaluator_name):
    """
        Get an evaluator instance from the registry based on the name of evaluator
        """
    evaluator_cls = self._registry.get(evaluator_name)
    if evaluator_cls is None:
        raise MlflowException(f'Could not find a registered model evaluator for: {evaluator_name}. Currently registered evaluator names are: {list(self._registry.keys())}')
    return evaluator_cls()