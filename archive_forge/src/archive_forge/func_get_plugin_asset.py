import abc
from tensorflow.python.framework import ops
def get_plugin_asset(plugin_asset_cls, graph=None):
    """Acquire singleton PluginAsset instance from a graph.

  PluginAssets are always singletons, and are stored in tf Graph collections.
  This way, they can be defined anywhere the graph is being constructed, and
  if the same plugin is configured at many different points, the user can always
  modify the same instance.

  Args:
    plugin_asset_cls: The PluginAsset class
    graph: (optional) The graph to retrieve the instance from. If not specified,
      the default graph is used.

  Returns:
    An instance of the plugin_asset_class

  Raises:
    ValueError: If we have a plugin name collision, or if we unexpectedly find
      the wrong number of items in a collection.
  """
    if graph is None:
        graph = ops.get_default_graph()
    if not plugin_asset_cls.plugin_name:
        raise ValueError('Class %s has no plugin_name' % plugin_asset_cls.__name__)
    name = _PLUGIN_ASSET_PREFIX + plugin_asset_cls.plugin_name
    container = graph.get_collection(name)
    if container:
        if len(container) != 1:
            raise ValueError('Collection for %s had %d items, expected 1' % (name, len(container)))
        instance = container[0]
        if not isinstance(instance, plugin_asset_cls):
            raise ValueError('Plugin name collision between classes %s and %s' % (plugin_asset_cls.__name__, instance.__class__.__name__))
    else:
        instance = plugin_asset_cls()
        graph.add_to_collection(name, instance)
        graph.add_to_collection(_PLUGIN_ASSET_PREFIX, plugin_asset_cls.plugin_name)
    return instance