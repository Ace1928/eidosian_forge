import pickle
import re
from debian.deprecation import function_deprecated_by
def filter_packages_copy(self, filter_data):
    """
        Return a collection with only those packages that match a
        filter, with a copy of the tagsets of this one.  The filter
        will match on the package.
        """
    res = DB()
    db = {}
    for pkg in filter(filter_data, self.db.keys()):
        db[pkg] = self.db[pkg].copy()
    res.db = db
    res.rdb = reverse(db)
    return res