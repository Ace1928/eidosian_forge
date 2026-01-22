import paste.util.threadinglocal as threadinglocal
def make_registry_manager(app, global_conf):
    return RegistryManager(app)