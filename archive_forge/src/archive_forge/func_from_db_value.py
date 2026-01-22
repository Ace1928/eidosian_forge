import base64
import pickle
from django.db import models
from django.utils import encoding
import jsonpickle
import oauth2client
def from_db_value(self, value, expression, connection, context):
    """Overrides ``models.Field`` method. This converts the value
        returned from the database to an instance of this class.
        """
    return self.to_python(value)