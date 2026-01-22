from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_sqlalchemy_json(required: bool=False):
    """
    Ensures that `sqlalchemy_json` is available
    """
    global sqlalchemy_json, _sqlalchemy_json_available
    if not _sqlalchemy_json_available:
        resolve_missing('sqlalchemy_json', required=required)
        import sqlalchemy_json
        _sqlalchemy_json_available = True
        globals()['sqlalchemy_json'] = sqlalchemy_json