from parlai.core.teachers import DialogTeacher
from .build import build
import json
import os
def label_candidates(self):
    return self.cands