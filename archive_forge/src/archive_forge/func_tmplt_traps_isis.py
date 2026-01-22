from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_traps_isis(config_data):
    isis = config_data.get('traps', {}).get('isis', {})
    command = 'snmp-server traps isis'
    if isis.get('all'):
        command += ' all'
    else:
        if isis.get('adjacency_change'):
            command += ' adjacency-change'
        if isis.get('area_mismatch'):
            command += ' area-mismatch'
        if isis.get('attempt_to_exceed_max_sequence'):
            command += ' attempt-to-exceed-max-sequence'
        if isis.get('authentication_failure'):
            command += ' authentication-failure'
        if isis.get('authentication_type_failure'):
            command += ' authentication-type-failure'
        if isis.get('corrupted_lsp_detected'):
            command += ' corrupted-lsp-detected'
        if isis.get('database_overload'):
            command += ' database-overload'
        if isis.get('id_len_mismatch'):
            command += ' id-len-mismatch'
        if isis.get('lsp_error_detected'):
            command += ' lsp-error-detected'
        if isis.get('lsp_too_large_to_propagate'):
            command += ' lsp-too-large-to-propagate'
        if isis.get('manual_address_drops'):
            command += ' manual-address-drops'
        if isis.get('max_area_addresses_mismatch'):
            command += ' max-area-addresses-mismatch'
        if isis.get('orig_lsp_buff_size_mismatch'):
            command += ' orig-lsp-buff-size-mismatch'
        if isis.get('version_skew'):
            command += ' version-skew'
        if isis.get('own_lsp_purge'):
            command += ' own-lsp-purge'
        if isis.get('rejected_adjacency'):
            command += ' rejected-adjacency'
        if isis.get('protocols_supported_mismatch'):
            command += ' protocols-supported-mismatch'
        if isis.get('sequence_number_skip'):
            command += ' sequence-number-skip'
    return command