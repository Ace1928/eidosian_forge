from django.contrib.gis.geos.geometry import GEOSGeometry, hex_regex, wkt_regex
def fromstr(string, **kwargs):
    """Given a string value, return a GEOSGeometry object."""
    return GEOSGeometry(string, **kwargs)