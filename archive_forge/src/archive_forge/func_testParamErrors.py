from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def testParamErrors(self):
    for uri in ('/paramerrors/one_positional?param1=foo', '/paramerrors/one_positional_args?param1=foo', '/paramerrors/one_positional_args/foo', '/paramerrors/one_positional_args/foo/bar/baz', '/paramerrors/one_positional_args_kwargs?param1=foo&param2=bar', '/paramerrors/one_positional_args_kwargs/foo?param2=bar&param3=baz', '/paramerrors/one_positional_args_kwargs/foo/bar/baz?param2=bar&param3=baz', '/paramerrors/one_positional_kwargs?param1=foo&param2=bar&param3=baz', '/paramerrors/one_positional_kwargs/foo?param4=foo&param2=bar&param3=baz', '/paramerrors/no_positional', '/paramerrors/no_positional_args/foo', '/paramerrors/no_positional_args/foo/bar/baz', '/paramerrors/no_positional_args_kwargs?param1=foo&param2=bar', '/paramerrors/no_positional_args_kwargs/foo?param2=bar', '/paramerrors/no_positional_args_kwargs/foo/bar/baz?param2=bar&param3=baz', '/paramerrors/no_positional_kwargs?param1=foo&param2=bar', '/paramerrors/callable_object'):
        self.getPage(uri)
        self.assertStatus(200)
    error_msgs = ['Missing parameters', 'Nothing matches the given URI', 'Multiple values for parameters', 'Unexpected query string parameters', 'Unexpected body parameters', 'Invalid path in Request-URI', 'Illegal #fragment in Request-URI']
    for uri, error_idx in (('invalid/path/without/leading/slash', 5), ('/valid/path#invalid=fragment', 6)):
        self.getPage(uri)
        self.assertStatus(400)
        self.assertInBody(error_msgs[error_idx])
    for uri, msg in (('/paramerrors/one_positional', error_msgs[0]), ('/paramerrors/one_positional?foo=foo', error_msgs[0]), ('/paramerrors/one_positional/foo/bar/baz', error_msgs[1]), ('/paramerrors/one_positional/foo?param1=foo', error_msgs[2]), ('/paramerrors/one_positional/foo?param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo?param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo/bar/baz?param2=foo', error_msgs[3]), ('/paramerrors/one_positional_args_kwargs/foo/bar/baz?param1=bar&param3=baz', error_msgs[2]), ('/paramerrors/one_positional_kwargs/foo?param1=foo&param2=bar&param3=baz', error_msgs[2]), ('/paramerrors/no_positional/boo', error_msgs[1]), ('/paramerrors/no_positional?param1=foo', error_msgs[3]), ('/paramerrors/no_positional_args/boo?param1=foo', error_msgs[3]), ('/paramerrors/no_positional_kwargs/boo?param1=foo', error_msgs[1]), ('/paramerrors/callable_object?param1=foo', error_msgs[3]), ('/paramerrors/callable_object/boo', error_msgs[1])):
        for show_mismatched_params in (True, False):
            cherrypy.config.update({'request.show_mismatched_params': show_mismatched_params})
            self.getPage(uri)
            self.assertStatus(404)
            if show_mismatched_params:
                self.assertInBody(msg)
            else:
                self.assertInBody('Not Found')
    for uri, body, msg in (('/paramerrors/one_positional/foo', 'param1=foo', error_msgs[2]), ('/paramerrors/one_positional/foo', 'param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo', 'param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo/bar/baz', 'param2=foo', error_msgs[4]), ('/paramerrors/one_positional_args_kwargs/foo/bar/baz', 'param1=bar&param3=baz', error_msgs[2]), ('/paramerrors/one_positional_kwargs/foo', 'param1=foo&param2=bar&param3=baz', error_msgs[2]), ('/paramerrors/no_positional', 'param1=foo', error_msgs[4]), ('/paramerrors/no_positional_args/boo', 'param1=foo', error_msgs[4]), ('/paramerrors/callable_object', 'param1=foo', error_msgs[4])):
        for show_mismatched_params in (True, False):
            cherrypy.config.update({'request.show_mismatched_params': show_mismatched_params})
            self.getPage(uri, method='POST', body=body)
            self.assertStatus(400)
            if show_mismatched_params:
                self.assertInBody(msg)
            else:
                self.assertInBody('400 Bad')
    for uri, body, msg in (('/paramerrors/one_positional?param2=foo', 'param1=foo', error_msgs[3]), ('/paramerrors/one_positional/foo/bar', 'param2=foo', error_msgs[1]), ('/paramerrors/one_positional_args/foo/bar?param2=foo', 'param3=foo', error_msgs[3]), ('/paramerrors/one_positional_kwargs/foo/bar', 'param2=bar&param3=baz', error_msgs[1]), ('/paramerrors/no_positional?param1=foo', 'param2=foo', error_msgs[3]), ('/paramerrors/no_positional_args/boo?param2=foo', 'param1=foo', error_msgs[3]), ('/paramerrors/callable_object?param2=bar', 'param1=foo', error_msgs[3])):
        for show_mismatched_params in (True, False):
            cherrypy.config.update({'request.show_mismatched_params': show_mismatched_params})
            self.getPage(uri, method='POST', body=body)
            self.assertStatus(404)
            if show_mismatched_params:
                self.assertInBody(msg)
            else:
                self.assertInBody('Not Found')
    for uri in ('/paramerrors/raise_type_error', '/paramerrors/raise_type_error_with_default_param?x=0', '/paramerrors/raise_type_error_with_default_param?x=0&y=0', '/paramerrors/raise_type_error_decorated'):
        self.getPage(uri, method='GET')
        self.assertStatus(500)
        self.assertTrue('Client Error', self.body)