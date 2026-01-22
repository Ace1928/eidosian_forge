import socket
import geoip2.database
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_ipv46_address
from django.utils._os import to_path
from .resources import City, Country
def lon_lat(self, query):
    """Return a tuple of the (longitude, latitude) for the given query."""
    return self.coords(query)