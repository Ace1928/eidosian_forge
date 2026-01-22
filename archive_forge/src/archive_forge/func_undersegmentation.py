import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def undersegmentation(self):
    return np.mean([e.undersegmentation for e in self.eval_objects])