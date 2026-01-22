import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
def update_qos_allocation(self, consumer_uuid, alloc_diff):
    """Update allocation for QoS minimum bandwidth consumer

        :param consumer_uuid: The uuid of the consumer, in case of bound port
                              owned by a VM, the VM uuid.
        :param alloc_diff: A dict which contains RP UUIDs as keys and
                           corresponding fields to update for the allocation
                           under the given resource provider.
        """
    for i in range(GENERATION_CONFLICT_RETRIES):
        body = self.list_allocations(consumer_uuid)
        if not body['allocations']:
            raise n_exc.PlacementAllocationRemoved(consumer=consumer_uuid)
        for rp_uuid, diff in alloc_diff.items():
            if rp_uuid not in body['allocations']:
                raise n_exc.PlacementAllocationRpNotExists(resource_provider=rp_uuid, consumer=consumer_uuid)
            for drctn, value in diff.items():
                orig_value = body['allocations'][rp_uuid]['resources'].get(drctn, 0)
                new_value = orig_value + value
                if new_value > 0:
                    body['allocations'][rp_uuid]['resources'][drctn] = new_value
                else:
                    resources = body['allocations'][rp_uuid]['resources']
                    resources.pop(drctn, None)
        body['allocations'] = {rp: alloc for rp, alloc in body['allocations'].items() if alloc.get('resources')}
        try:
            return self.update_allocation(consumer_uuid, body)
        except ks_exc.Conflict as e:
            resp = e.response.json()
            if resp['errors'][0]['code'] == 'placement.concurrent_update':
                continue
            raise
    raise n_exc.PlacementAllocationGenerationConflict(consumer=consumer_uuid)