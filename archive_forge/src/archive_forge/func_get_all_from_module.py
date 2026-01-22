from __future__ import annotations
from typing import Optional
def get_all_from_module(name):
    """
        Module level imports
        """
    qname = f'plotnine.{name}'
    comment = f'# {qname}\n'
    m = import_module(qname)
    return comment + comma_join((f'"{x}"' for x in sorted(m.__all__)))