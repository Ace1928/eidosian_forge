from django import shortcuts
import django.conf
from six import wraps
from six.moves.urllib import parse
from oauth2client.contrib import django_util
def oauth_enabled(decorated_function=None, scopes=None, **decorator_kwargs):
    """ Decorator to enable OAuth Credentials if authorized, and setup
    the oauth object on the request object to provide helper functions
    to start the flow otherwise.

    .. code-block:: python
       :caption: views.py
       :name: views_enabled3

       from oauth2client.django_util.decorators import oauth_enabled

       @oauth_enabled
       def optional_oauth2(request):
           if request.oauth.has_credentials():
               # this could be passed into a view
               # request.oauth.http is also initialized
               return HttpResponse("User email: {0}".format(
                                   request.oauth.credentials.id_token['email'])
           else:
               return HttpResponse('Here is an OAuth Authorize link:
               <a href="{0}">Authorize</a>'.format(
                   request.oauth.get_authorize_redirect()))


    Args:
        decorated_function: View function to decorate.
        scopes: Scopes to require, will default.
        decorator_kwargs: Can include ``return_url`` to specify the URL to
           return to after OAuth2 authorization is complete.

    Returns:
         The decorated view function.
    """

    def curry_wrapper(wrapped_function):

        @wraps(wrapped_function)
        def enabled_wrapper(request, *args, **kwargs):
            return_url = decorator_kwargs.pop('return_url', request.get_full_path())
            user_oauth = django_util.UserOAuth2(request, scopes, return_url)
            setattr(request, django_util.oauth2_settings.request_prefix, user_oauth)
            return wrapped_function(request, *args, **kwargs)
        return enabled_wrapper
    if decorated_function:
        return curry_wrapper(decorated_function)
    else:
        return curry_wrapper