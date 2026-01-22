import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def seventhsbass(self):
    return np.mean([e.seventhsbass for e in self.eval_objects])