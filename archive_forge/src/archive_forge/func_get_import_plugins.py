from oslo_config import cfg
from stevedore import named
def get_import_plugins(**kwargs):
    task_list = CONF.image_import_opts.image_import_plugins
    extensions = named.NamedExtensionManager('glance.image_import.plugins', names=task_list, name_order=True, invoke_on_load=True, invoke_kwds=kwargs)
    for extension in extensions.extensions:
        yield extension.obj