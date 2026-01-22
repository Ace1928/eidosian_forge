import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
@functools.lru_cache(maxsize=_OBSERVER_EXPRESSION_CACHE_MAXSIZE)
def compile_expr(expr):
    """ Compile an ObserverExpression to a list of ObserverGraphs.

    Parameters
    ----------
    expr : ObserverExpression

    Returns
    -------
    list of ObserverGraph
    """
    return expr._as_graphs()