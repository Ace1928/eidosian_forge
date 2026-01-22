import os.path
import cherrypy

Tutorial - The default method

Request handler objects can implement a method called "default" that
is called when no other suitable method/object could be found.
Essentially, if CherryPy2 can't find a matching request handler object
for the given request URI, it will use the default method of the object
located deepest on the URI path.

Using this mechanism you can easily simulate virtual URI structures
by parsing the extra URI string, which you can access through
cherrypy.request.virtualPath.

The application in this tutorial simulates an URI structure looking
like /users/<username>. Since the <username> bit will not be found (as
there are no matching methods), it is handled by the default method.
