import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def oversegmentation(self):
    return np.mean([e.oversegmentation for e in self.eval_objects])