from modin.config import Engine, IsExperimental, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.utils import _inherit_docstrings, get_current_execution
@classmethod
@_inherit_docstrings(factories.PandasOnRayFactory._read_sql_distributed)
def read_sql_distributed(cls, **kwargs):
    return cls.get_factory()._read_sql_distributed(**kwargs)