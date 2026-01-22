from typing import Any, Dict, Type
from sklearn.base import is_classifier, is_regressor
from triad import assert_or_throw
from triad.utils.convert import get_full_type_path, to_type
from tune.constants import SPACE_MODEL_NAME
from tune.concepts.space.spaces import Space
def sk_space(model: str, **params: Dict[str, Any]) -> Space:
    data = {SPACE_MODEL_NAME: to_sk_model_expr(model)}
    data.update(params)
    return Space(**data)