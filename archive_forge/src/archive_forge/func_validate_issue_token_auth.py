from keystone.common import validation
from keystone.common.validation import parameter_types
from keystone import exception
from keystone.i18n import _
def validate_issue_token_auth(auth=None):
    if auth is None:
        return
    validation.lazy_validate(token_issue, auth)
    user = auth['identity'].get('password', {}).get('user')
    if user is not None:
        if 'id' not in user and 'name' not in user:
            msg = _('Invalid input for field identity/password/user: id or name must be present.')
            raise exception.SchemaValidationError(detail=msg)
        domain = user.get('domain')
        if domain is not None:
            if 'id' not in domain and 'name' not in domain:
                msg = _('Invalid input for field identity/password/user/domain: id or name must be present.')
                raise exception.SchemaValidationError(detail=msg)
    scope = auth.get('scope')
    if scope is not None and isinstance(scope, dict):
        project = scope.get('project')
        if project is not None:
            if 'id' not in project and 'name' not in project:
                msg = _('Invalid input for field scope/project: id or name must be present.')
                raise exception.SchemaValidationError(detail=msg)
            domain = project.get('domain')
            if domain is not None:
                if 'id' not in domain and 'name' not in domain:
                    msg = _('Invalid input for field scope/project/domain: id or name must be present.')
                    raise exception.SchemaValidationError(detail=msg)
        domain = scope.get('domain')
        if domain is not None:
            if 'id' not in domain and 'name' not in domain:
                msg = _('Invalid input for field scope/domain: id or name must be present.')
                raise exception.SchemaValidationError(detail=msg)