import tempfile
from typing import Any, Dict, List, Tuple
from fs.base import FS as FSBase
from tensorflow import keras
from triad import FileSystem
from tune.concepts.space import to_template, TuningParametersTemplate
def compile_model(self, **add_kwargs: Any) -> keras.models.Model:
    params = dict(self.get_compile_params())
    params.update(add_kwargs)
    model = self.get_model()
    model.compile(**params)
    return model