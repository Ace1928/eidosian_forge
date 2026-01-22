from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.utils.cache import patch_vary_headers

    A view decorator that adds the specified headers to the Vary header of the
    response. Usage:

       @vary_on_headers('Cookie', 'Accept-language')
       def index(request):
           ...

    Note that the header names are not case-sensitive.
    