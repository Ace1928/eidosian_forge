from oslo_log import log as logging
def log_fail_msg(manager, entrypoint, exception):
    if hasattr(entrypoint, 'module'):
        LOG.warning('Encountered exception while loading %(module_name)s: "%(message)s". Not using %(name)s.', {'module_name': entrypoint.module, 'message': getattr(exception, 'message', str(exception)), 'name': entrypoint.name})
    else:
        LOG.warning('Encountered exception: "%(message)s". Not using %(name)s.', {'message': getattr(exception, 'message', str(exception)), 'name': entrypoint.name})