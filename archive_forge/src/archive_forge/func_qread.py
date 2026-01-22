import pickle
import re
from debian.deprecation import function_deprecated_by
def qread(self, file):
    """Quickly read the data from a pickled file"""
    self.db = pickle.load(file)
    self.rdb = pickle.load(file)