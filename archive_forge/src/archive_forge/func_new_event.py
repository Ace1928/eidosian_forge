from pycadf import cadftype
from pycadf import event
def new_event(self, eventType=cadftype.EVENTTYPE_ACTIVITY, **kwargs):
    """Create new event

        :param eventType: eventType of event. Defaults to 'activity'
        """
    event_val = event.Event(**kwargs)
    if not cadftype.is_valid_eventType(eventType):
        raise ValueError(ERROR_UNKNOWN_EVENTTYPE)
    event_val.eventType = eventType
    return event_val