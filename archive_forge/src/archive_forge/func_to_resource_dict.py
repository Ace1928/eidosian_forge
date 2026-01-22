import logging
import sys
from collections import namedtuple
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
def to_resource_dict(self):
    """Returns a dict suitable to pass to raylet initialization.

        This renames num_cpus / num_gpus to "CPU" / "GPU",
        translates memory from bytes into 100MB memory units, and checks types.
        """
    assert self.resolved()
    resources = dict(self.resources, CPU=self.num_cpus, GPU=self.num_gpus, memory=int(self.memory), object_store_memory=int(self.object_store_memory))
    resources = {resource_label: resource_quantity for resource_label, resource_quantity in resources.items() if resource_quantity != 0}
    for resource_label, resource_quantity in resources.items():
        assert isinstance(resource_quantity, int) or isinstance(resource_quantity, float), f'{resource_label} ({type(resource_quantity)}): {resource_quantity}'
        if isinstance(resource_quantity, float) and (not resource_quantity.is_integer()):
            raise ValueError("Resource quantities must all be whole numbers. Violated by resource '{}' in {}.".format(resource_label, resources))
        if resource_quantity < 0:
            raise ValueError("Resource quantities must be nonnegative. Violated by resource '{}' in {}.".format(resource_label, resources))
        if resource_quantity > ray_constants.MAX_RESOURCE_QUANTITY:
            raise ValueError("Resource quantities must be at most {}. Violated by resource '{}' in {}.".format(ray_constants.MAX_RESOURCE_QUANTITY, resource_label, resources))
    return resources