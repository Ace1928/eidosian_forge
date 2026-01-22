from ._auth_context import ContextKey
def context_with_operations(ctx, ops):
    """ Returns a context(AuthContext) which is associated with all the given
    operations (list of string). It will be based on the auth context
    passed in as ctx.

    An allow caveat will succeed only if one of the allowed operations is in
    ops; a deny caveat will succeed only if none of the denied operations are
    in ops.
    """
    return ctx.with_value(OP_KEY, ops)