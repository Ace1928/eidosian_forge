import flask
from werkzeug import exceptions as werkzeug_exceptions
from keystone import exception
from keystone.i18n import _
from keystone.server.flask import common as ks_flask_common
def json_body_before_request():
    """Enforce JSON Request Body."""
    if not flask.request.get_data():
        return None
    elif flask.request.path and flask.request.path.startswith('/v3/OS-OAUTH2/'):
        return None
    try:
        if flask.request.is_json or flask.request.headers.get('Content-Type', '') == '':
            json_decoded = flask.request.get_json(force=True)
            if not isinstance(json_decoded, dict):
                raise werkzeug_exceptions.BadRequest(_('resulting JSON load was not a dict'))
        else:
            ks_flask_common.set_unenforced_ok()
            raise exception.ValidationError(attribute='application/json', target='Content-Type header')
    except werkzeug_exceptions.BadRequest:
        ks_flask_common.set_unenforced_ok()
        raise exception.ValidationError(attribute='valid JSON', target='request body')