from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
import json
import re
import csv
def get_title_dict(self, path):
    csv_path = os.path.join(path, 'movies_with_mentions.csv')
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            self.title_id_map['@' + row[0]] = remove_year_from_title(row[1])