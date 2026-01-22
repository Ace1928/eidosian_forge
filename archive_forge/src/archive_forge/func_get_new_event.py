import sys
def get_new_event(sa=None, bManualReset=True, bInitialState=True, objectName=None):
    return win32event.CreateEvent(sa, bManualReset, bInitialState, objectName)