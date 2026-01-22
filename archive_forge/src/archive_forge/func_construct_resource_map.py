import abc
import collections
import functools
import re
import uuid
import wsgiref.util
import flask
from flask import blueprints
import flask_restful
import flask_restful.utils
import http.client
from oslo_log import log
from oslo_log import versionutils
from oslo_serialization import jsonutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import driver_hints
from keystone.common import json_home
from keystone.common.rbac_enforcer import enforcer
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def construct_resource_map(resource, url, resource_kwargs, alternate_urls=None, rel=None, status=json_home.Status.STABLE, path_vars=None, resource_relation_func=_v3_resource_relation):
    """Construct the ResourceMap Named Tuple.

    :param resource: The flask-RESTful resource class implementing the methods
                     for the API.
    :type resource: :class:`ResourceMap`
    :param url: Flask-standard url route, all flask url routing rules apply.
                url variables will be passed to the Resource methods as
                arguments.
    :type url: str
    :param resource_kwargs: a dict of optional value(s) that can further modify
                            the handling of the routing.

                            * endpoint: endpoint name (defaults to
                                        :meth:`Resource.__name__.lower`
                                        Can be used to reference this route in
                                        :class:`fields.Url` fields (str)

                            * resource_class_args: args to be forwarded to the
                                                   constructor of the resource.
                                                   (tuple)

                            * resource_class_kwargs: kwargs to be forwarded to
                                                     the constructor of the
                                                     resource. (dict)

                            Additional keyword arguments not specified above
                            will be passed as-is to
                            :meth:`flask.Flask.add_url_rule`.
    :param alternate_urls: An iterable (list) of dictionaries containing urls
                           and associated json home REL data. Each element is
                           expected to be a dictionary with a 'url' key and an
                           optional 'json_home' key for a 'JsonHomeData' named
                           tuple  These urls will also map to the resource.
                           These are used to ensure API compatibility when a
                           "new" path is more correct for the API but old paths
                           must continue to work. Example:
                           `/auth/domains` being the new path for
                           `/OS-FEDERATION/domains`. The `OS-FEDERATION` part
                           would be listed as an alternate url. If a
                           'json_home' key is provided, the original path
                           with the new json_home data will be added to the
                           JSON Home Document.
    :type: iterable or None
    :param rel:
    :type rel: str or None
    :param status: JSON Home API Status, e.g. "STABLE"
    :type status: str
    :param path_vars: JSON Home Path Var Data (arguments)
    :type path_vars: dict or None
    :param resource_relation_func: function to build expected resource rel data
    :type resource_relation_func: callable
    :return:
    """
    if rel is not None:
        jh_data = construct_json_home_data(rel=rel, status=status, path_vars=path_vars, resource_relation_func=resource_relation_func)
    else:
        jh_data = None
    if not url.startswith('/'):
        url = '/%s' % url
    return ResourceMap(resource=resource, url=url, alternate_urls=alternate_urls, kwargs=resource_kwargs, json_home_data=jh_data)