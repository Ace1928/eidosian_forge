from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def pubnub_event_handler(block, data):
    """Retrieve reference on target event handler from application model.

    :type block:  Block
    :param block: Reference on block model from which reference on event
                  handlers should be fetched.
    :type data:   dict
    :param data:  Reference on dictionary which contain information about
                  event handler and whether it should be created or not.

    :rtype:  EventHandler
    :return: Reference on initialized and ready to use event handler model.
             'None' will be returned in case if there is no handler with
             specified name and no request to create it.
    """
    event_handler = block.event_handler(data['name'])
    changed_name = data.pop('changes').get('name') if 'changes' in data else None
    name = data.get('name') or changed_name
    channels = data.get('channels')
    event = data.get('event')
    code = _content_of_file_at_path(data.get('src'))
    state = data.get('state') or 'present'
    if event_handler is None and state == 'present':
        event_handler = EventHandler(name=name, channels=channels, event=event, code=code)
        block.add_event_handler(event_handler)
    if event_handler is not None and state == 'present':
        if name is not None:
            event_handler.name = name
        if channels is not None:
            event_handler.channels = channels
        if event is not None:
            event_handler.event = event
        if code is not None:
            event_handler.code = code
    return event_handler