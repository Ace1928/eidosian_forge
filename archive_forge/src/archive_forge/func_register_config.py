from paste.registry import RegistryManager, StackedObjectProxy
def register_config(environ, start_response):
    popped_config = environ.get(environ_key, no_config)
    current_config = environ[environ_key] = config.copy()
    environ['paste.registry'].register(dispatching_config, current_config)
    try:
        app_iter = application(environ, start_response)
    finally:
        if popped_config is no_config:
            environ.pop(environ_key, None)
        else:
            environ[environ_key] = popped_config
    return app_iter