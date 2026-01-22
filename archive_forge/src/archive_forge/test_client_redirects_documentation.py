import http.client as http
import eventlet.patcher
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils

            Handles all requests to the application.
            