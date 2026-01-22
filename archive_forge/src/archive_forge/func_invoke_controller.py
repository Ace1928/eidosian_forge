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
def invoke_controller(self, controller, args, kwargs, state):
    """
        The main request handler for Pecan applications.
        """
    cfg = _cfg(controller)
    content_types = cfg.get('content_types', {})
    req = state.request
    resp = state.response
    pecan_state = req.pecan
    argspec = getargspec(controller)
    keys = kwargs.keys()
    for key in keys:
        if key not in argspec.args and (not argspec.keywords):
            kwargs.pop(key)
    result = controller(*args, **kwargs)
    if result is response:
        return
    elif isinstance(result, WebObResponse):
        state.response = result
        return
    raw_namespace = result
    template = content_types.get(pecan_state['content_type'])
    template = pecan_state.get('override_template', template)
    if template is None and cfg['explicit_content_type'] is False:
        if self.default_renderer == 'json':
            template = 'json'
    pecan_state['content_type'] = pecan_state.get('override_content_type', pecan_state['content_type'])
    if template:
        if template == 'json':
            pecan_state['content_type'] = 'application/json'
        result = self.render(template, result)
    if req.environ.get('paste.testing'):
        testing_variables = req.environ['paste.testing_variables']
        testing_variables['namespace'] = raw_namespace
        testing_variables['template_name'] = template
        testing_variables['controller_output'] = result
    if result and isinstance(result, str):
        resp.text = result
    elif result:
        resp.body = result
    if pecan_state['content_type']:
        resp.content_type = pecan_state['content_type']