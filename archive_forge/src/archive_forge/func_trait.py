import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
def trait(self, name, notify=True, optional=False):
    """ Create a new expression for observing a trait with the exact
        name given.

        Events emitted (if any) will be instances of
        :class:`~traits.observation.events.TraitChangeEvent`.

        Parameters
        ----------
        name : str
            Name of the trait to match.
        notify : bool, optional
            Whether to notify for changes. Default is to notify.
        optional : bool, optional
            If true, skip this observer if the requested trait is not found.
            Default is false, and an error will be raised if the requested
            trait is not found.

        Returns
        -------
        new_expression : ObserverExpression
        """
    return self.then(trait(name=name, notify=notify, optional=optional))