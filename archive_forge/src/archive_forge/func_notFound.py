from typing import cast
from twisted.web import http
from twisted.web.iweb import IRenderable, IRequest
from twisted.web.resource import IResource, Resource
from twisted.web.template import renderElement, tags
def notFound(brief: str='No Such Resource', message: str='Sorry. No luck finding that resource.') -> IResource:
    """
    Generate an L{IResource} with a 404 Not Found status code.

    @see: L{twisted.web.pages.errorPage}

    @param brief: A short string displayed as the page title.

    @param brief: A longer string displayed in the page body.

    @returns: An L{IResource}
    """
    return _ErrorPage(http.NOT_FOUND, brief, message)