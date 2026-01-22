from django.contrib.syndication.views import Feed as BaseFeed
from django.utils.feedgenerator import Atom1Feed, Rss201rev2Feed
def georss_coords(self, coords):
    """
        In GeoRSS coordinate pairs are ordered by lat/lon and separated by
        a single white space. Given a tuple of coordinates, return a string
        GeoRSS representation.
        """
    return ' '.join(('%f %f' % (coord[1], coord[0]) for coord in coords))