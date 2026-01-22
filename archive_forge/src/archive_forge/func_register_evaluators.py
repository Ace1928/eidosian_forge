import warnings
import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.utils.import_hooks import register_post_import_hook
def register_evaluators(module):
    from mlflow.models.evaluation.default_evaluator import DefaultEvaluator
    module._model_evaluation_registry.register('default', DefaultEvaluator)
    module._model_evaluation_registry.register_entrypoints()