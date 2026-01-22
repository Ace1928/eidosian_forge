import inspect
import unittest
from traits.observation import expression
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._observer_graph import ObserverGraph
def test_dict_items_method_optional(self):
    expr = expression.dict_items().dict_items(optional=True)
    expected = [create_graph(DictItemObserver(notify=True, optional=False), DictItemObserver(notify=True, optional=True))]
    actual = expr._as_graphs()
    self.assertEqual(actual, expected)