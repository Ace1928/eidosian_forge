from __future__ import annotations
import asyncio
import json
import logging
import os
import typing as ty
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from socket import gaierror
from jupyter_events import EventLogger
from tornado import web
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPResponse
from traitlets import (
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
def load_connection_args(self, **kwargs):
    """Merges the static args relative to the connection, with the given keyword arguments.  If static
        args have yet to be initialized, we'll do that here.

        """
    if len(self._connection_args) == 0:
        self.init_connection_args()
    prev_auth_token = self.auth_token
    if self.auth_token is not None:
        try:
            self.auth_token = self.gateway_token_renewer.get_token(self.auth_header_key, self.auth_scheme, self.auth_token)
        except Exception as ex:
            self.log.error(f"An exception occurred attempting to renew the Gateway authorization token using an instance of class '{self.gateway_token_renewer_class}'.  The request will proceed using the current token value.  Exception was: {ex}")
            self.auth_token = prev_auth_token
    for arg, value in self._connection_args.items():
        if arg == 'headers':
            given_value = kwargs.setdefault(arg, {})
            if isinstance(given_value, dict):
                given_value.update(value)
                given_value.update({f'{self.auth_header_key}': f'{self.auth_scheme} {self.auth_token}'})
        else:
            kwargs[arg] = value
    if self.accept_cookies:
        self._update_cookie_header(kwargs)
    return kwargs