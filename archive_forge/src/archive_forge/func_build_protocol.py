import abc
@abc.abstractmethod
def build_protocol(self, socket):
    """Create an instance of a subclass of Protocol.

        Override this method to alter how Protocol instances get created.
        """
    raise NotImplementedError()