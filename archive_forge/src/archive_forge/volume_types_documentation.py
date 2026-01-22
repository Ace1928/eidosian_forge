from urllib import parse
from cinderclient.apiclient import base as common_base
from cinderclient import base
Update the name and/or description for a volume type.

        :param volume_type: The ID of the :class:`VolumeType` to update.
        :param name: Descriptive name of the volume type.
        :param description: Description of the volume type.
        :rtype: :class:`VolumeType`
        