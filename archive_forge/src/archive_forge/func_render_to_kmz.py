import zipfile
from io import BytesIO
from django.conf import settings
from django.http import HttpResponse
from django.template import loader
def render_to_kmz(*args, **kwargs):
    """
    Compress the KML content and return as KMZ (using the correct
    MIME type).
    """
    return HttpResponse(compress_kml(loader.render_to_string(*args, **kwargs)), content_type='application/vnd.google-earth.kmz')