from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (

    Update multiple records. If an operation failed, raise a DNSAPIException.

    @param api: A ZoneRecordAPI instance
    @param provider_information: A ProviderInformation object.
    @param options: A object compatible with ModuleOptionProvider that gives access to the module/plugin
                    options.
    @param zone_id: Zone ID to apply changes to
    @param records_to_delete: Optional list of DNS records to delete (DNSRecord)
    @param records_to_change: Optional list of DNS records to change (DNSRecord)
    @param records_to_create: Optional list of DNS records to create (DNSRecord)
    @param bulk_threshold: Minimum number of changes for using the bulk API instead of the regular API
    @param stop_early_on_errors: If set to ``True``, try to stop changes after the first error happens.
                                 This might only work on some APIs.
    @return A tuple (changed, errors, success) where ``changed`` is a boolean which indicates whether a
            change was made, ``errors`` is a list of ``DNSAPIError`` instances for the errors occurred,
            and ``success`` is a dictionary with three lists ``success['deleted']``,
            ``success['changed']`` and ``success['created']``, which list all records that were deleted,
            changed and created, respectively.
    