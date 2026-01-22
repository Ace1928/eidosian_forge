import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
@property
def sevenths(self):
    return np.mean([e.sevenths for e in self.eval_objects])