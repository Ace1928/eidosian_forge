from matplotlib import _api, backend_tools, cbook, widgets
def message_event(self, message, sender=None):
    """Emit a `ToolManagerMessageEvent`."""
    if sender is None:
        sender = self
    s = 'tool_message_event'
    event = ToolManagerMessageEvent(s, sender, message)
    self._callbacks.process(s, event)