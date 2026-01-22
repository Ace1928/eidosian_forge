import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
 Dumps the object content in a csv file format.

            Parameters
            ----------
            dest: string
                Destination file path.
            delimiter: char
                Character to separate values in the csv file.
        