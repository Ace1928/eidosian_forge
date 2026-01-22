from monascaclient.common import monasca_manager
def list_dimension_names(self, **kwargs):
    """Get a list of metric dimension names."""
    return self._list('/dimensions/names', **kwargs)