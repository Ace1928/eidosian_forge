from typing import Any, Type, Dict
from triad.utils.convert import get_full_type_path, to_type
from tune.concepts.space.parameters import TuningParametersTemplate
from tune_tensorflow.spec import KerasTrainingSpec
from tune import Space
from tune.constants import SPACE_MODEL_NAME
def keras_space(model: Any, **params: Any) -> Space:
    expr = to_keras_spec_expr(model)
    _TYPE_DICT[expr] = to_keras_spec(model)
    data = {SPACE_MODEL_NAME: expr}
    data.update(params)
    return Space(**data)