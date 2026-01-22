from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_nfs_share(self, id=None, name=None):
    """ Get the nfs export

        :return: nfs_export object if nfs exists else None
        :rtype: UnityNfsShare or None
        """
    try:
        if not id and (not name):
            msg = 'Please give nfs id/name'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        id_or_name = id if id else name
        LOG.info('Getting nfs export: %s', id_or_name)
        if id:
            if self.is_given_nfs_for_fs:
                nfs = self.unity.get_nfs_share(_id=id, filesystem=self.fs_obj)
            elif self.is_given_nfs_for_fs is False:
                nfs = self.unity.get_nfs_share(_id=id, snap=self.snap_obj)
            else:
                nfs = self.unity.get_nfs_share(_id=id)
        elif self.is_given_nfs_for_fs:
            nfs = self.unity.get_nfs_share(name=name, filesystem=self.fs_obj)
        elif self.is_given_nfs_for_fs is False:
            nfs = self.unity.get_nfs_share(name=name, snap=self.snap_obj)
        else:
            nfs = self.unity.get_nfs_share(name=name)
        if isinstance(nfs, utils.UnityNfsShareList):
            nfs_list = nfs
            LOG.info('Multiple nfs export with same name: %s found', id_or_name)
            if self.nas_obj:
                for n in nfs_list:
                    if n.filesystem.nas_server == self.nas_obj:
                        return n
                msg = 'Multiple nfs share with same name: %s found. Given nas server is not correct. Please check'
            else:
                msg = 'Multiple nfs share with same name: %s found. Please give nas server'
        elif nfs and nfs.existed:
            if self.nas_obj and nfs.filesystem.nas_server != self.nas_obj:
                msg = 'nfs found but nas details given is incorrect'
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            LOG.info('Successfully got nfs share for: %s', id_or_name)
            return nfs
        elif nfs and (not nfs.existed):
            msg = 'Please check incorrect nfs id is given'
        else:
            msg = 'Failed to get nfs share: %s' % id_or_name
    except utils.UnityResourceNotFoundError as e:
        msg = 'NFS share: %(id_or_name)s not found error: %(err)s' % {'id_or_name': id_or_name, 'err': str(e)}
        LOG.info(str(msg))
        return None
    except utils.HTTPClientError as e:
        if e.http_status == 401:
            msg = 'Failed to get nfs share: %s due to incorrect username/password error: %s' % (id_or_name, str(e))
        else:
            msg = 'Failed to get nfs share: %s error: %s' % (id_or_name, str(e))
    except utils.StoropsConnectTimeoutError as e:
        msg = 'Failed to get nfs share: %s check unispherehost IP: %s error: %s' % (id_or_name, self.module.params['nfs_export_id'], str(e))
    except Exception as e:
        msg = 'Failed to get nfs share: %s error: %s' % (id_or_name, str(e))
    LOG.error(msg)
    self.module.fail_json(msg=msg)