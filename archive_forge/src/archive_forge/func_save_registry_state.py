import paste.util.threadinglocal as threadinglocal
def save_registry_state(self, environ):
    """Save the state of this request's Registry (if it hasn't already been
        saved) to the saved_registry_states dict, keyed by the request's unique
        identifier"""
    registry = environ.get('paste.registry')
    if not registry or not len(registry.reglist) or self.get_request_id(environ) in self.saved_registry_states:
        return
    self.saved_registry_states[self.get_request_id(environ)] = (registry, registry.reglist[:])
    for reglist in registry.reglist:
        for stacked, obj in reglist.values():
            self.enable_restoration(stacked)