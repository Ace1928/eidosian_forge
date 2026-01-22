from io import BytesIO
from ...bzr.smart import medium
from ...transport import chroot, get_transport
from ...urlutils import local_path_to_url
def make_app(root, prefix, path_var='REQUEST_URI', readonly=True, load_plugins=True, enable_logging=True):
    """Convenience function to construct a WSGI bzr smart server.

    :param root: a local path that requests will be relative to.
    :param prefix: See RelpathSetter.
    :param path_var: See RelpathSetter.
    """
    local_url = local_path_to_url(root)
    if readonly:
        base_transport = get_transport('readonly+' + local_url)
    else:
        base_transport = get_transport(local_url)
    if load_plugins:
        from ...plugin import load_plugins
        load_plugins()
    if enable_logging:
        import breezy.trace
        breezy.trace.enable_default_logging()
    app = SmartWSGIApp(base_transport, prefix)
    app = RelpathSetter(app, '', path_var)
    return app