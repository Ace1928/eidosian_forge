import threading
from routes.mapper import Mapper
from routes.util import redirect_to, url_for, URLGenerator
def load_wsgi_environ(self, environ):
    """
        Load the protocol/server info from the environ and store it.
        Also, match the incoming URL if there's already a mapper, and
        store the resulting match dict in mapper_dict.
        """
    if 'HTTPS' in environ or environ.get('wsgi.url_scheme') == 'https' or environ.get('HTTP_X_FORWARDED_PROTO') == 'https':
        self.__shared_state.protocol = 'https'
    else:
        self.__shared_state.protocol = 'http'
    try:
        self.mapper.environ = environ
    except AttributeError:
        pass
    try:
        if 'PATH_INFO' in environ:
            mapper = self.mapper
            path = environ['PATH_INFO']
            result = mapper.routematch(path)
            if result is not None:
                self.__shared_state.mapper_dict = result[0]
                self.__shared_state.route = result[1]
            else:
                self.__shared_state.mapper_dict = None
                self.__shared_state.route = None
    except AttributeError:
        pass
    if 'HTTP_X_FORWARDED_HOST' in environ:
        self.__shared_state.host = environ['HTTP_X_FORWARDED_HOST'].split(', ', 1)[0]
    elif 'HTTP_HOST' in environ:
        self.__shared_state.host = environ['HTTP_HOST']
    else:
        self.__shared_state.host = environ['SERVER_NAME']
        if environ['wsgi.url_scheme'] == 'https':
            if environ['SERVER_PORT'] != '443':
                self.__shared_state.host += ':' + environ['SERVER_PORT']
        elif environ['SERVER_PORT'] != '80':
            self.__shared_state.host += ':' + environ['SERVER_PORT']