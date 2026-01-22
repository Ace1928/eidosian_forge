import flask
from flask import request
import http.client
from oslo_serialization import jsonutils
from keystone.common import json_home
import keystone.conf
from keystone.server import flask as ks_flask
def v3_mime_type_best_match():
    if not request.accept_mimetypes:
        return MimeTypes.JSON
    return request.accept_mimetypes.best_match([MimeTypes.JSON, MimeTypes.JSON_HOME])