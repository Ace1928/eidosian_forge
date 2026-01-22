from tensorboard import auth as auth_lib
def set_in_environ(environ, ctx):
    """Set the `RequestContext` in a WSGI environment.

    After `set_in_environ(e, ctx)`, `from_environ(e) is ctx`. The input
    environment is mutated.

    Args:
      environ: A WSGI environment to update.
      ctx: A new `RequestContext` value.
    """
    environ[_WSGI_KEY] = ctx