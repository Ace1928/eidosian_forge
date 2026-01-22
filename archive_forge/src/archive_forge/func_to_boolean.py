import uuid
from boto.compat import urllib
from boto.resultset import ResultSet
def to_boolean(self, value, true_value='true'):
    if value == true_value:
        return True
    else:
        return False