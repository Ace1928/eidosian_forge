import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def dumps_csv(self, delimiter=','):
    str_buffer = StringIO()
    csv_writer = csv.writer(str_buffer, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for entry in self.as_list():
        csv_writer.writerow(str(entry))
    return str_buffer.getvalue()