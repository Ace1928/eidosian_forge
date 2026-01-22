import asyncio
import concurrent.futures
import threading
from wsgiref.validate import validator
from tornado.routing import RuleRouter
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.wsgi import WSGIContainer
def respond_plain(self, start_response):
    status = '200 OK'
    response_headers = [('Content-Type', 'text/plain')]
    start_response(status, response_headers)