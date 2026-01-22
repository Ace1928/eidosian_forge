import abc
import collections
import enum
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.common import style  # pylint: disable=unused-import
from grpc.framework.foundation import future  # pylint: disable=unused-import
from grpc.framework.foundation import stream  # pylint: disable=unused-import
@abc.abstractmethod
def protocol_context(self):
    """Accesses a custom object specified by an implementation provider.

        Returns:
          A value specified by the provider of a Face interface implementation
            affording custom state and behavior.
        """
    raise NotImplementedError()