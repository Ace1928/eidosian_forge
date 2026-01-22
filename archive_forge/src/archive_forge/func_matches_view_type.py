from mlflow.entities.view_type import ViewType
from mlflow.exceptions import MlflowException
@classmethod
def matches_view_type(cls, view_type, lifecycle_stage):
    if not cls.is_valid(lifecycle_stage):
        raise MlflowException(f"Invalid lifecycle stage '{lifecycle_stage}'")
    if view_type == ViewType.ALL:
        return True
    elif view_type == ViewType.ACTIVE_ONLY:
        return lifecycle_stage == LifecycleStage.ACTIVE
    elif view_type == ViewType.DELETED_ONLY:
        return lifecycle_stage == LifecycleStage.DELETED
    else:
        raise MlflowException(f"Invalid view type '{view_type}'")