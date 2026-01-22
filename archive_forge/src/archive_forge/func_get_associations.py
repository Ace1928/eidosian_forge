from cinderclient.apiclient import base as common_base
from cinderclient import base
def get_associations(self, qos_specs):
    """Get associated entities of a qos specs.

        :param qos_specs: The id of the :class: `QoSSpecs`
        :return: a list of entities that associated with specific qos specs.
        """
    return self._list('/qos-specs/%s/associations' % base.getid(qos_specs), 'qos_associations')