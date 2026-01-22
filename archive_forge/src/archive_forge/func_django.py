from django.contrib.gis.gdal.error import GDALException
@property
def django(self):
    """Return the Django GeometryField for this OGR Type."""
    s = self.name.replace('25D', '')
    if s in ('LinearRing', 'None'):
        return None
    elif s == 'Unknown':
        s = 'Geometry'
    elif s == 'PointZ':
        s = 'Point'
    return s + 'Field'