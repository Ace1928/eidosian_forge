from django.contrib.syndication.views import Feed as BaseFeed
from django.utils.feedgenerator import Atom1Feed, Rss201rev2Feed
def rss_attributes(self):
    attrs = super().rss_attributes()
    attrs['xmlns:geo'] = 'http://www.w3.org/2003/01/geo/wgs84_pos#'
    return attrs