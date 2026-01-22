from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def cancel_target_job(self, job, ex_client=None, ex_target=None):
    """
        Cancel a backup job on a target

        :param job: Backup target job to cancel.  If it is ``None``
                    ex_client and ex_target must be set
        :type  job: Instance of :class:`BackupTargetJob` or ``None``

        :param ex_client: Client of the job to cancel.
                          Not necessary if job is specified.
                          DimensionData only has 1 job per client
        :type  ex_client: Instance of :class:`DimensionDataBackupClient`
                          or ``str``

        :param ex_target: Target to cancel a job from.
                          Not necessary if job is specified.
        :type  ex_target: Instance of :class:`BackupTarget` or ``str``

        :rtype: ``bool``
        """
    if job is None:
        if ex_client is None or ex_target is None:
            raise ValueError('Either job or ex_client and ex_target have to be set')
        server_id = self._target_to_target_address(ex_target)
        client_id = self._client_to_client_id(ex_client)
    else:
        server_id = job.target.address
        client_id = job.extra['clientId']
    response = self.connection.request_with_orgId_api_1('server/{}/backup/client/{}?cancelJob'.format(server_id, client_id), method='GET').object
    response_code = findtext(response, 'result', GENERAL_NS)
    return response_code in ['IN_PROGRESS', 'SUCCESS']