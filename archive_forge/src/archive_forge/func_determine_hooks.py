from inspect import Arguments
from itertools import chain, tee
from mimetypes import guess_type, add_type
from os.path import splitext
import logging
import operator
import sys
import types
from webob import (Request as WebObRequest, Response as WebObResponse, exc,
from webob.multidict import NestedMultiDict
from .compat import urlparse, izip, is_bound_method as ismethod
from .jsonify import encode as dumps
from .secure import handle_security
from .templating import RendererFactory
from .routing import lookup_controller, NonCanonicalPath
from .util import _cfg, getargspec
from .middleware.recursive import ForwardRequestException
def determine_hooks(self, controller=None):
    """
        Determines the hooks to be run, in which order.

        :param controller: If specified, includes hooks for a specific
                           controller.
        """
    controller_hooks = []
    if controller:
        controller_hooks = _cfg(controller).get('hooks', [])
        if controller_hooks:
            return list(sorted(chain(controller_hooks, self.hooks), key=operator.attrgetter('priority')))
    return self.hooks