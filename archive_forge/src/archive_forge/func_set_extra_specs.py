from openstack import exceptions
from openstack import resource
from openstack import utils
def set_extra_specs(self, session, **extra_specs):
    """Update extra specs.

        This call will replace only the extra_specs with the same keys
        given here.  Other keys will not be modified.

        :param session: The session to use for making this request.
        :param kwargs extra_specs: Key/value extra_specs pairs to be update on
            this volume type. All keys and values.
        :returns: The updated extra specs.
        """
    if not extra_specs:
        return dict()
    result = self._extra_specs(session.post, extra_specs=extra_specs)
    return result['extra_specs']