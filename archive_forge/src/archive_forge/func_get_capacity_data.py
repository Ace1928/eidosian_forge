from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
def get_capacity_data(self, catalog):
    """Generate a list of devices/OS versions & corresponding capacity info.

    Args:
      catalog: Android or iOS catalog

    Returns:
      The list of device models, versions, and capacity info we want to have
      printed later. Obsolete (unsupported) devices, versions, and entries
      missing capacity info are filtered out.
    """
    capacity_data = []
    for model in catalog.models:
        for version_info in model.perVersionInfo:
            if version_info.versionId not in model.supportedVersionIds:
                continue
            capacity_data.append(CapacityEntry(model=model.id, name=model.name, version=version_info.versionId, capacity=self.capacity_messages[version_info.deviceCapacity]))
    return capacity_data