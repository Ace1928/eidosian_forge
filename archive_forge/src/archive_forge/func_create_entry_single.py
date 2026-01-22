from parlai.core.teachers import FixedDialogTeacher
from .build import build, RESOURCES
import os
import json
def create_entry_single(episode):
    entry = []
    for key in self.existing_keys:
        if key in episode:
            entry.append(str(episode[key]))
        else:
            entry.append('N/A')
    return entry