import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
def middleware_exceptions(method):

    @functools.wraps(method)
    def _inner(self, request):
        try:
            return method(self, request)
        except exception.Error as e:
            LOG.warning(e)
            return render_exception(e, request=request, user_locale=best_match_language(request))
        except TypeError as e:
            LOG.exception(e)
            return render_exception(exception.ValidationError(e), request=request, user_locale=best_match_language(request))
        except Exception as e:
            LOG.exception(e)
            return render_exception(exception.UnexpectedError(exception=e), request=request, user_locale=best_match_language(request))
    return _inner