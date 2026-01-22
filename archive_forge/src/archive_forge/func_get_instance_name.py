import dataclasses
from typing import Any
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.mesh import plugin_data_pb2
def get_instance_name(name, content_type):
    """Returns a unique instance name for a given summary related to the
    mesh."""
    return '%s_%s' % (name, plugin_data_pb2.MeshPluginData.ContentType.Name(content_type))