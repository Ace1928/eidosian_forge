import base64
import os.path
import uuid
from .. import __version__
def bugreport(app=None):
    try:
        import celery
        import humanize
        import tornado
        app = app or celery.Celery()
        return 'flower   -> flower:%s tornado:%s humanize:%s%s' % (__version__, tornado.version, getattr(humanize, '__version__', None) or getattr(humanize, 'VERSION'), app.bugreport())
    except (ImportError, AttributeError) as e:
        return f"Error when generating bug report: {e}. Have you installed correct versions of Flower's dependencies?"