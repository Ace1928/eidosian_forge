import pickle
import re
from debian.deprecation import function_deprecated_by
def package_count(self):
    """Return the number of packages"""
    return len(self.db)