from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
def track_cascade_events(descriptor, prop):
    """Establish event listeners on object attributes which handle
    cascade-on-set/append.

    """
    key = prop.key

    def append(state, item, initiator, **kw):
        if item is None:
            return
        sess = state.session
        if sess:
            if sess._warn_on_events:
                sess._flush_warning('collection append')
            prop = state.manager.mapper._props[key]
            item_state = attributes.instance_state(item)
            if prop._cascade.save_update and key == initiator.key and (not sess._contains_state(item_state)):
                sess._save_or_update_state(item_state)
        return item

    def remove(state, item, initiator, **kw):
        if item is None:
            return
        sess = state.session
        prop = state.manager.mapper._props[key]
        if sess and sess._warn_on_events:
            sess._flush_warning('collection remove' if prop.uselist else 'related attribute delete')
        if item is not None and item is not attributes.NEVER_SET and (item is not attributes.PASSIVE_NO_RESULT) and prop._cascade.delete_orphan:
            item_state = attributes.instance_state(item)
            if prop.mapper._is_orphan(item_state):
                if sess and item_state in sess._new:
                    sess.expunge(item)
                else:
                    item_state._orphaned_outside_of_session = True

    def set_(state, newvalue, oldvalue, initiator, **kw):
        if oldvalue is newvalue:
            return newvalue
        sess = state.session
        if sess:
            if sess._warn_on_events:
                sess._flush_warning('related attribute set')
            prop = state.manager.mapper._props[key]
            if newvalue is not None:
                newvalue_state = attributes.instance_state(newvalue)
                if prop._cascade.save_update and key == initiator.key and (not sess._contains_state(newvalue_state)):
                    sess._save_or_update_state(newvalue_state)
            if oldvalue is not None and oldvalue is not attributes.NEVER_SET and (oldvalue is not attributes.PASSIVE_NO_RESULT) and prop._cascade.delete_orphan:
                oldvalue_state = attributes.instance_state(oldvalue)
                if oldvalue_state in sess._new and prop.mapper._is_orphan(oldvalue_state):
                    sess.expunge(oldvalue)
        return newvalue
    event.listen(descriptor, 'append_wo_mutation', append, raw=True, include_key=True)
    event.listen(descriptor, 'append', append, raw=True, retval=True, include_key=True)
    event.listen(descriptor, 'remove', remove, raw=True, retval=True, include_key=True)
    event.listen(descriptor, 'set', set_, raw=True, retval=True, include_key=True)