from django.contrib.gis.db.backends.postgis.pgraster import to_pgraster
from django.contrib.gis.geos import GEOSGeometry
from django.db.backends.postgresql.psycopg_any import sql
def getquoted(self):
    """
        Return a properly quoted string for use in PostgreSQL/PostGIS.
        """
    if self.is_geometry:
        return b'%s(%s)' % (b'ST_GeogFromWKB' if self.geography else b'ST_GeomFromEWKB', sql.quote(self.ewkb).encode())
    else:
        return b"'%s'::raster" % self.ewkb.hex().encode()