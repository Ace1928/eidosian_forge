from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def volume_id_exists(self, volume_id):
    """
            Return volume_id if volume exists for given volume_id

            :param volume_id: volume ID
            :type volume_id: int
            :return: Volume ID if found, None if not found
            :rtype: int
        """
    volume_list = self.elem_connect.list_volumes(volume_ids=[volume_id])
    for volume in volume_list.volumes:
        if volume.volume_id == volume_id:
            if str(volume.delete_time) == '':
                return volume.volume_id
    return None