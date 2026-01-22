from typing import Any, Type, Dict
from triad.utils.convert import get_full_type_path, to_type
from tune.concepts.space.parameters import TuningParametersTemplate
from tune_tensorflow.spec import KerasTrainingSpec
from tune import Space
from tune.constants import SPACE_MODEL_NAME
def to_keras_spec(obj: Any) -> Type[KerasTrainingSpec]:
    if isinstance(obj, str) and obj in _TYPE_DICT:
        return _TYPE_DICT[obj]
    return to_type(obj, KerasTrainingSpec)