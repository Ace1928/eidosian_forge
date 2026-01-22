from oslo_cache import core
from oslo_config import cfg
from heat.common.i18n import _
def register_cache_configurations(conf):
    """Register all configurations required for oslo.cache.

    The procedure registers all configurations required for oslo.cache.
    It should be called before configuring of cache region

    :param conf: instance of heat configuration
    :returns: updated heat configuration
    """
    core.configure(conf)
    conf.register_group(constraint_cache_group)
    conf.register_opts(constraint_cache_opts, group=constraint_cache_group)
    conf.register_group(extension_cache_group)
    conf.register_opts(extension_cache_opts, group=extension_cache_group)
    conf.register_group(find_cache_group)
    conf.register_opts(find_cache_opts, group=find_cache_group)
    return conf