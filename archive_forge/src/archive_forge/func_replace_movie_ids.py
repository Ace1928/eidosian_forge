from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
import re
import csv
def replace_movie_ids(id_string, id_map):
    pattern = '@\\d+'
    return re.sub(pattern, lambda s: id_map[s.group()], id_string)