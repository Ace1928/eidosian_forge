from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def require_sql(required: bool=True, require_sqlalchemy: bool=True, require_psycopg2: bool=False, require_asyncpg: bool=False, require_sqlalchemy_json: bool=True):
    """
    Wrapper for `resolve_sqlalchemy` that can be used as a decorator
    """

    def decorator(func):
        return require_missing_wrapper(resolver=resolve_sql, func=func, required=required, require_sqlalchemy=require_sqlalchemy, require_psycopg2=require_psycopg2, require_asyncpg=require_asyncpg, require_sqlalchemy_json=require_sqlalchemy_json)
    return decorator