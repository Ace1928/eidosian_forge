import os
import sys
import linecache
import re
import inspect
def format_asyncio_info():
    """ Returns a formatted string of the asyncio info.
    This can be useful in determining what's going on in the asyncio event
    loop system, especially when used in conjunction with the asyncio hub.
    """
    import asyncio
    tasks = asyncio.all_tasks()
    result = ['TASKS:']
    result.append(repr(tasks))
    result.append(f'EVENTLOOP: {asyncio.events.get_event_loop()}')
    return os.linesep.join(result)