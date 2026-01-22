import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
def test_UUIDv4_parameter_log_format(log_tracker, monkeypatch, server):
    """Test rendering of UUID4 within access log."""
    monkeypatch.setattr('cherrypy._cplogging.LogManager.access_log_format', '{i}')
    log_tracker.markLog()
    host = webtest.interface(webtest.WebCase.HOST)
    port = webtest.WebCase.PORT
    requests.get('http://%s:%s/as_string' % (host, port))
    log_tracker.assertValidUUIDv4()