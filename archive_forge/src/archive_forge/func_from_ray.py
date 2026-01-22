from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
@_inherit_docstrings(factories.BaseFactory._from_ray)
def from_ray(cls, ray_obj):
    return cls.get_factory()._from_ray(ray_obj)