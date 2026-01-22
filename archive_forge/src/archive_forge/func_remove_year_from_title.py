from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
import re
import csv
def remove_year_from_title(title):
    matches = re.finditer('\\s\\(', title)
    indices = [m.start(0) for m in matches]
    if indices:
        title_end = indices[-1]
        return title[:title_end]
    else:
        return title