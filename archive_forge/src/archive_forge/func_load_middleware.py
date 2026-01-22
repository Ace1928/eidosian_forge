from urllib.parse import urlparse
from urllib.request import url2pathname
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.staticfiles import utils
from django.contrib.staticfiles.views import serve
from django.core.handlers.asgi import ASGIHandler
from django.core.handlers.exception import response_for_exception
from django.core.handlers.wsgi import WSGIHandler, get_path_info
from django.http import Http404
def load_middleware(self):
    pass