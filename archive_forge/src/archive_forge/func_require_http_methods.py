import datetime
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpResponseNotAllowed
from django.middleware.http import ConditionalGetMiddleware
from django.utils import timezone
from django.utils.cache import get_conditional_response
from django.utils.decorators import decorator_from_middleware
from django.utils.http import http_date, quote_etag
from django.utils.log import log_response
def require_http_methods(request_method_list):
    """
    Decorator to make a view only accept particular request methods.  Usage::

        @require_http_methods(["GET", "POST"])
        def my_view(request):
            # I can assume now that only GET or POST requests make it this far
            # ...

    Note that request methods should be in uppercase.
    """

    def decorator(func):
        if iscoroutinefunction(func):

            @wraps(func)
            async def inner(request, *args, **kwargs):
                if request.method not in request_method_list:
                    response = HttpResponseNotAllowed(request_method_list)
                    log_response('Method Not Allowed (%s): %s', request.method, request.path, response=response, request=request)
                    return response
                return await func(request, *args, **kwargs)
        else:

            @wraps(func)
            def inner(request, *args, **kwargs):
                if request.method not in request_method_list:
                    response = HttpResponseNotAllowed(request_method_list)
                    log_response('Method Not Allowed (%s): %s', request.method, request.path, response=response, request=request)
                    return response
                return func(request, *args, **kwargs)
        return inner
    return decorator