from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from ..orm.collections import collection
from ..orm.collections import collection_adapter
def ordering_list(attr: str, count_from: Optional[int]=None, ordering_func: Optional[OrderingFunc]=None, reorder_on_append: bool=False) -> Callable[[], OrderingList]:
    """Prepares an :class:`OrderingList` factory for use in mapper definitions.

    Returns an object suitable for use as an argument to a Mapper
    relationship's ``collection_class`` option.  e.g.::

        from sqlalchemy.ext.orderinglist import ordering_list

        class Slide(Base):
            __tablename__ = 'slide'

            id = Column(Integer, primary_key=True)
            name = Column(String)

            bullets = relationship("Bullet", order_by="Bullet.position",
                                    collection_class=ordering_list('position'))

    :param attr:
      Name of the mapped attribute to use for storage and retrieval of
      ordering information

    :param count_from:
      Set up an integer-based ordering, starting at ``count_from``.  For
      example, ``ordering_list('pos', count_from=1)`` would create a 1-based
      list in SQL, storing the value in the 'pos' column.  Ignored if
      ``ordering_func`` is supplied.

    Additional arguments are passed to the :class:`.OrderingList` constructor.

    """
    kw = _unsugar_count_from(count_from=count_from, ordering_func=ordering_func, reorder_on_append=reorder_on_append)
    return lambda: OrderingList(attr, **kw)