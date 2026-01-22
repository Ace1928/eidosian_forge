import warnings
from io import StringIO
def factory_(app):

    class EnvironForward(ForwardRequestExceptionMiddleware):

        def __call__(self, environ_, start_response):
            return self.app(environ, start_response)
    return EnvironForward(app)