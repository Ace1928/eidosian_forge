from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def operate_ospf_info(self):
    """operate ospf info"""
    config_route_id_xml = ''
    vrf = self.get_exist_vrf()
    if vrf is None:
        vrf = '_public_'
    description = self.get_exist_description()
    if description is None:
        description = ''
    bandwidth_reference = self.get_exist_bandwidth()
    if bandwidth_reference is None:
        bandwidth_reference = '100'
    lsa_in_interval = self.get_exist_lsa_a_interval()
    if lsa_in_interval is None:
        lsa_in_interval = ''
    lsa_arrival_max_interval = self.get_exist_lsa_a_max_interval()
    if lsa_arrival_max_interval is None:
        lsa_arrival_max_interval = '1000'
    lsa_arrival_start_interval = self.get_exist_lsa_a_start_interval()
    if lsa_arrival_start_interval is None:
        lsa_arrival_start_interval = '500'
    lsa_arrival_hold_interval = self.get_exist_lsa_a_hold_interval()
    if lsa_arrival_hold_interval is None:
        lsa_arrival_hold_interval = '500'
    lsa_originate_interval = self.getexistlsaointerval()
    if lsa_originate_interval is None:
        lsa_originate_interval = '5'
    lsa_originate_max_interval = self.getexistlsaomaxinterval()
    if lsa_originate_max_interval is None:
        lsa_originate_max_interval = '5000'
    lsa_originate_start_interval = self.getexistlsaostartinterval()
    if lsa_originate_start_interval is None:
        lsa_originate_start_interval = '500'
    lsa_originate_hold_interval = self.getexistlsaoholdinterval()
    if lsa_originate_hold_interval is None:
        lsa_originate_hold_interval = '1000'
    spf_interval = self.get_exist_spf_interval()
    if spf_interval is None:
        spf_interval = ''
    spf_interval_milli = self.get_exist_spf_milli_interval()
    if spf_interval_milli is None:
        spf_interval_milli = ''
    spf_max_interval = self.get_exist_spf_max_interval()
    if spf_max_interval is None:
        spf_max_interval = '5000'
    spf_start_interval = self.get_exist_spf_start_interval()
    if spf_start_interval is None:
        spf_start_interval = '50'
    spf_hold_interval = self.get_exist_spf_hold_interval()
    if spf_hold_interval is None:
        spf_hold_interval = '200'
    if self.route_id:
        if self.state == 'present':
            if self.route_id != self.get_exist_route():
                self.route_id_changed = True
                config_route_id_xml = CE_NC_CREATE_ROUTE_ID % self.route_id
        else:
            if self.route_id != self.get_exist_route():
                self.module.fail_json(msg='Error: The route id %s is not exist.' % self.route_id)
            self.route_id_changed = True
            configxmlstr = CE_NC_DELETE_OSPF % (self.ospf, self.get_exist_route(), self.get_exist_vrf())
            conf_str = build_config_xml(configxmlstr)
            recv_xml = set_nc_config(self.module, conf_str)
            self.check_response(recv_xml, 'OPERATE_VRF_AF')
            self.changed = True
            return
    if self.vrf != '_public_':
        if self.state == 'present':
            if self.vrf != self.get_exist_vrf():
                self.vrf_changed = True
                vrf = self.vrf
        else:
            if self.vrf != self.get_exist_vrf():
                self.module.fail_json(msg='Error: The vrf %s is not exist.' % self.vrf)
            self.vrf_changed = True
            configxmlstr = CE_NC_DELETE_OSPF % (self.ospf, self.get_exist_route(), self.get_exist_vrf())
            conf_str = build_config_xml(configxmlstr)
            recv_xml = set_nc_config(self.module, conf_str)
            self.check_response(recv_xml, 'OPERATE_VRF_AF')
            self.changed = True
            return
    if self.bandwidth:
        if self.state == 'present':
            if self.bandwidth != self.get_exist_bandwidth():
                self.bandwidth_changed = True
                bandwidth_reference = self.bandwidth
        else:
            if self.bandwidth != self.get_exist_bandwidth():
                self.module.fail_json(msg='Error: The bandwidth %s is not exist.' % self.bandwidth)
            if self.get_exist_bandwidth() != '100':
                self.bandwidth_changed = True
                bandwidth_reference = '100'
    if self.description:
        if self.state == 'present':
            if self.description != self.get_exist_description():
                self.description_changed = True
                description = self.description
        else:
            if self.description != self.get_exist_description():
                self.module.fail_json(msg='Error: The description %s is not exist.' % self.description)
            self.description_changed = True
            description = ''
    if self.lsaalflag is False:
        lsa_in_interval = ''
        if self.state == 'present':
            if self.lsaamaxinterval:
                if self.lsaamaxinterval != self.get_exist_lsa_a_max_interval():
                    self.lsa_arrival_changed = True
                    lsa_arrival_max_interval = self.lsaamaxinterval
            if self.lsaastartinterval:
                if self.lsaastartinterval != self.get_exist_lsa_a_start_interval():
                    self.lsa_arrival_changed = True
                    lsa_arrival_start_interval = self.lsaastartinterval
            if self.lsaaholdinterval:
                if self.lsaaholdinterval != self.get_exist_lsa_a_hold_interval():
                    self.lsa_arrival_changed = True
                    lsa_arrival_hold_interval = self.lsaaholdinterval
        else:
            if self.lsaamaxinterval:
                if self.lsaamaxinterval != self.get_exist_lsa_a_max_interval():
                    self.module.fail_json(msg='Error: The lsaamaxinterval %s is not exist.' % self.lsaamaxinterval)
                if self.get_exist_lsa_a_max_interval() != '1000':
                    lsa_arrival_max_interval = '1000'
                    self.lsa_arrival_changed = True
            if self.lsaastartinterval:
                if self.lsaastartinterval != self.get_exist_lsa_a_start_interval():
                    self.module.fail_json(msg='Error: The lsaastartinterval %s is not exist.' % self.lsaastartinterval)
                if self.get_exist_lsa_a_start_interval() != '500':
                    lsa_arrival_start_interval = '500'
                    self.lsa_arrival_changed = True
            if self.lsaaholdinterval:
                if self.lsaaholdinterval != self.get_exist_lsa_a_hold_interval():
                    self.module.fail_json(msg='Error: The lsaaholdinterval %s is not exist.' % self.lsaaholdinterval)
                if self.get_exist_lsa_a_hold_interval() != '500':
                    lsa_arrival_hold_interval = '500'
                    self.lsa_arrival_changed = True
    elif self.state == 'present':
        lsaalflag = 'false'
        if self.lsaalflag is True:
            lsaalflag = 'true'
        if lsaalflag != self.get_exist_lsa_a_interval_flag():
            self.lsa_arrival_changed = True
            if self.lsaainterval is None:
                self.module.fail_json(msg='Error: The lsaainterval is not supplied.')
            else:
                lsa_in_interval = self.lsaainterval
        elif self.lsaainterval:
            if self.lsaainterval != self.get_exist_lsa_a_interval():
                self.lsa_arrival_changed = True
                lsa_in_interval = self.lsaainterval
    elif self.lsaainterval:
        if self.lsaainterval != self.get_exist_lsa_a_interval():
            self.module.fail_json(msg='Error: The lsaainterval %s is not exist.' % self.lsaainterval)
        self.lsaalflag = False
        lsa_in_interval = ''
        self.lsa_arrival_changed = True
    if self.lsaointervalflag is False:
        if self.state == 'present':
            if self.lsaomaxinterval:
                if self.lsaomaxinterval != self.getexistlsaomaxinterval():
                    self.lsa_originate_changed = True
                    lsa_originate_max_interval = self.lsaomaxinterval
            if self.lsaostartinterval:
                if self.lsaostartinterval != self.getexistlsaostartinterval():
                    self.lsa_originate_changed = True
                    lsa_originate_start_interval = self.lsaostartinterval
            if self.lsaoholdinterval:
                if self.lsaoholdinterval != self.getexistlsaoholdinterval():
                    self.lsa_originate_changed = True
                    lsa_originate_hold_interval = self.lsaoholdinterval
            if self.lsaointerval:
                if self.lsaointerval != self.getexistlsaointerval():
                    self.lsa_originate_changed = True
                    lsa_originate_interval = self.lsaointerval
        else:
            if self.lsaomaxinterval:
                if self.lsaomaxinterval != self.getexistlsaomaxinterval():
                    self.module.fail_json(msg='Error: The lsaomaxinterval %s is not exist.' % self.lsaomaxinterval)
                if self.getexistlsaomaxinterval() != '5000':
                    lsa_originate_max_interval = '5000'
                    self.lsa_originate_changed = True
            if self.lsaostartinterval:
                if self.lsaostartinterval != self.getexistlsaostartinterval():
                    self.module.fail_json(msg='Error: The lsaostartinterval %s is not exist.' % self.lsaostartinterval)
                if self.getexistlsaostartinterval() != '500':
                    lsa_originate_start_interval = '500'
                    self.lsa_originate_changed = True
            if self.lsaoholdinterval:
                if self.lsaoholdinterval != self.getexistlsaoholdinterval():
                    self.module.fail_json(msg='Error: The lsaoholdinterval %s is not exist.' % self.lsaoholdinterval)
                if self.getexistlsaoholdinterval() != '1000':
                    lsa_originate_hold_interval = '1000'
                    self.lsa_originate_changed = True
            if self.lsaointerval:
                if self.lsaointerval != self.getexistlsaointerval():
                    self.module.fail_json(msg='Error: The lsaointerval %s is not exist.' % self.lsaointerval)
                if self.getexistlsaointerval() != '5':
                    lsa_originate_interval = '5'
                    self.lsa_originate_changed = True
    elif self.state == 'present':
        if self.getexistlsaointerval_flag() != 'true':
            self.lsa_originate_changed = True
            lsa_originate_interval = '5'
            lsa_originate_max_interval = '5000'
            lsa_originate_start_interval = '500'
            lsa_originate_hold_interval = '1000'
    elif self.getexistlsaointerval_flag() == 'true':
        self.lsaointervalflag = False
        self.lsa_originate_changed = True
    if self.spfintervaltype != self.get_exist_spf_interval_type():
        self.spf_changed = True
    if self.spfintervaltype == 'timer':
        if self.spfinterval:
            if self.state == 'present':
                if self.spfinterval != self.get_exist_spf_interval():
                    self.spf_changed = True
                    spf_interval = self.spfinterval
                    spf_interval_milli = ''
            else:
                if self.spfinterval != self.get_exist_spf_interval():
                    self.module.fail_json(msg='Error: The spfinterval %s is not exist.' % self.spfinterval)
                self.spfintervaltype = 'intelligent-timer'
                spf_interval = ''
                self.spf_changed = True
    if self.spfintervaltype == 'millisecond':
        if self.spfintervalmi:
            if self.state == 'present':
                if self.spfintervalmi != self.get_exist_spf_milli_interval():
                    self.spf_changed = True
                    spf_interval_milli = self.spfintervalmi
                    spf_interval = ''
            else:
                if self.spfintervalmi != self.get_exist_spf_milli_interval():
                    self.module.fail_json(msg='Error: The spfintervalmi %s is not exist.' % self.spfintervalmi)
                self.spfintervaltype = 'intelligent-timer'
                spf_interval_milli = ''
                self.spf_changed = True
    if self.spfintervaltype == 'intelligent-timer':
        spf_interval = ''
        spf_interval_milli = ''
        if self.spfmaxinterval:
            if self.state == 'present':
                if self.spfmaxinterval != self.get_exist_spf_max_interval():
                    self.spf_changed = True
                    spf_max_interval = self.spfmaxinterval
            else:
                if self.spfmaxinterval != self.get_exist_spf_max_interval():
                    self.module.fail_json(msg='Error: The spfmaxinterval %s is not exist.' % self.spfmaxinterval)
                if self.get_exist_spf_max_interval() != '5000':
                    self.spf_changed = True
                    spf_max_interval = '5000'
        if self.spfstartinterval:
            if self.state == 'present':
                if self.spfstartinterval != self.get_exist_spf_start_interval():
                    self.spf_changed = True
                    spf_start_interval = self.spfstartinterval
            else:
                if self.spfstartinterval != self.get_exist_spf_start_interval():
                    self.module.fail_json(msg='Error: The spfstartinterval %s is not exist.' % self.spfstartinterval)
                if self.get_exist_spf_start_interval() != '50':
                    self.spf_changed = True
                    spf_start_interval = '50'
        if self.spfholdinterval:
            if self.state == 'present':
                if self.spfholdinterval != self.get_exist_spf_hold_interval():
                    self.spf_changed = True
                    spf_hold_interval = self.spfholdinterval
            else:
                if self.spfholdinterval != self.get_exist_spf_hold_interval():
                    self.module.fail_json(msg='Error: The spfholdinterval %s is not exist.' % self.spfholdinterval)
                if self.get_exist_spf_hold_interval() != '200':
                    self.spf_changed = True
                    spf_hold_interval = '200'
    if not self.description_changed and (not self.vrf_changed) and (not self.lsa_arrival_changed) and (not self.lsa_originate_changed) and (not self.spf_changed) and (not self.route_id_changed) and (not self.bandwidth_changed):
        self.changed = False
        return
    else:
        self.changed = True
    lsaointervalflag = 'false'
    lsaalflag = 'false'
    if self.lsaointervalflag is True:
        lsaointervalflag = 'true'
    if self.lsaalflag is True:
        lsaalflag = 'true'
    configxmlstr = CE_NC_CREATE_OSPF_VRF % (self.ospf, config_route_id_xml, vrf, description, bandwidth_reference, lsaalflag, lsa_in_interval, lsa_arrival_max_interval, lsa_arrival_start_interval, lsa_arrival_hold_interval, lsaointervalflag, lsa_originate_interval, lsa_originate_max_interval, lsa_originate_start_interval, lsa_originate_hold_interval, self.spfintervaltype, spf_interval, spf_interval_milli, spf_max_interval, spf_start_interval, spf_hold_interval)
    conf_str = build_config_xml(configxmlstr)
    recv_xml = set_nc_config(self.module, conf_str)
    self.check_response(recv_xml, 'OPERATE_VRF_AF')