from matplotlib import _api, backend_tools, cbook, widgets
def remove_tool(self, name):
    """
        Remove tool named *name*.

        Parameters
        ----------
        name : str
            Name of the tool.
        """
    tool = self.get_tool(name)
    if getattr(tool, 'toggled', False):
        self.trigger_tool(tool, 'toolmanager')
    self._remove_keys(name)
    event = ToolEvent('tool_removed_event', self, tool)
    self._callbacks.process(event.name, event)
    del self._tools[name]