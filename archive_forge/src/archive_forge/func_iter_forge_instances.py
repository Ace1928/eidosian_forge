import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
def iter_forge_instances(forge: Optional[Type[Forge]]=None):
    """Iterate over all known forge instances.

    :return: Iterator over Forge instances
    """
    if forge is None:
        forge_clses = [forge_cls for name, forge_cls in forges.items()]
    else:
        forge_clses = [forge]
    for forge_cls in forge_clses:
        yield from forge_cls.iter_instances()