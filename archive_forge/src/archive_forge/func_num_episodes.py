from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
def num_episodes(self):
    return self.num_examples()