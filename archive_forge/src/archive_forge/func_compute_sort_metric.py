import tempfile
from typing import Any, Dict, List, Tuple
from fs.base import FS as FSBase
from tensorflow import keras
from triad import FileSystem
from tune.concepts.space import to_template, TuningParametersTemplate
def compute_sort_metric(self, **add_kwargs: Any) -> float:
    metric = self.get_fit_metric(self.fit(**add_kwargs))
    return self.generate_sort_metric(metric)