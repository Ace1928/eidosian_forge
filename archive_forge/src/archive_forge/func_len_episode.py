import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
def len_episode(self, ep):
    d = self.data[ep]
    first_speaker = d['dialog'][0]['speaker'].lower()
    if self.speaker_label != 'both' and self.speaker_label in first_speaker:
        return (len(d['dialog']) - 1) // 2
    return len(d['dialog']) // 2