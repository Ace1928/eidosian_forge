from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from bokeh.document import Document
from bokeh.server.contexts import BokehSessionContext, _RequestProxy
from bokeh.server.session import ServerSession
from bokeh.settings import settings
from bokeh.util.token import generate_jwt_token, generate_session_id
def generate_session(application, request=None, payload=None, initialize=True):
    secret_key = settings.secret_key_bytes()
    sign_sessions = settings.sign_sessions()
    session_id = generate_session_id(secret_key=secret_key, signed=sign_sessions)
    payload = payload or {}
    token = generate_jwt_token(session_id, secret_key=secret_key, signed=sign_sessions, extra_payload=payload)
    doc = Document()
    session_context = BokehSessionContext(session_id, None, doc)
    session_context._request = _RequestProxy(request, arguments=payload.get('arguments'), cookies=payload.get('cookies'), headers=payload.get('headers'))
    session_context._token = token
    doc._session_context = lambda: session_context
    if initialize:
        application.initialize_document(doc)
    callbacks = doc.callbacks._session_callbacks
    doc.callbacks._session_callbacks = set()
    session = ServerSessionStub(session_id, doc, io_loop=None, token=token)
    doc.callbacks._session_callbacks = callbacks
    return session