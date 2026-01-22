from django.contrib.gis import gdal
@property
def spheroid(self):
    """Return the spheroid name for this spatial reference."""
    return self.srs['spheroid']