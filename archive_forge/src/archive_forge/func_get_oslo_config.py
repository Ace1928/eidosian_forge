def get_oslo_config():
    """Runtime load the oslo.config object.

    In performance optimization of openstackclient it was determined that even
    optimistically loading oslo.config if available had a performance cost.
    Given that we used to only raise the ImportError when the function was
    called also attempt to do the import to do everything at runtime.
    """
    global cfg
    if not cfg:
        try:
            from oslo_config import cfg
        except ImportError:
            cfg = _NOT_FOUND
    if cfg is _NOT_FOUND:
        raise ImportError("oslo.config is not an automatic dependency of keystoneauth. If you wish to use oslo.config you need to import it into your application's requirements file. ")
    return cfg