import itertools
import logging
import warnings
import os_ken.base.app_manager
from os_ken.lib import hub
from os_ken import utils
from os_ken.controller import ofp_event
from os_ken.controller.controller import OpenFlowController
from os_ken.controller.handler import set_ev_handler
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER,\
from os_ken.ofproto import ofproto_parser
@set_ev_handler(ofp_event.EventOFPHello, HANDSHAKE_DISPATCHER)
def hello_handler(self, ev):
    self.logger.debug('hello ev %s', ev)
    msg = ev.msg
    datapath = msg.datapath
    elements = getattr(msg, 'elements', None)
    if elements:
        switch_versions = set()
        for version in itertools.chain.from_iterable((element.versions for element in elements)):
            switch_versions.add(version)
        usable_versions = switch_versions & set(datapath.supported_ofp_version)
        negotiated_versions = set((version for version in switch_versions if version <= max(datapath.supported_ofp_version)))
        if negotiated_versions and (not usable_versions):
            error_desc = 'no compatible version found: switch versions %s controller version 0x%x, the negotiated version is 0x%x, but no usable version found. If possible, set the switch to use one of OF version %s' % (switch_versions, max(datapath.supported_ofp_version), max(negotiated_versions), sorted(datapath.supported_ofp_version))
            self._hello_failed(datapath, error_desc)
            return
        if negotiated_versions and usable_versions and (max(negotiated_versions) != max(usable_versions)):
            error_desc = 'no compatible version found: switch versions 0x%x controller version 0x%x, the negotiated version is %s but found usable %s. If possible, set the switch to use one of OF version %s' % (max(switch_versions), max(datapath.supported_ofp_version), sorted(negotiated_versions), sorted(usable_versions), sorted(usable_versions))
            self._hello_failed(datapath, error_desc)
            return
    else:
        usable_versions = set((version for version in datapath.supported_ofp_version if version <= msg.version))
        if usable_versions and max(usable_versions) != min(msg.version, datapath.ofproto.OFP_VERSION):
            version = max(usable_versions)
            error_desc = 'no compatible version found: switch 0x%x controller 0x%x, but found usable 0x%x. If possible, set the switch to use OF version 0x%x' % (msg.version, datapath.ofproto.OFP_VERSION, version, version)
            self._hello_failed(datapath, error_desc)
            return
    if not usable_versions:
        error_desc = 'unsupported version 0x%x. If possible, set the switch to use one of the versions %s' % (msg.version, sorted(datapath.supported_ofp_version))
        self._hello_failed(datapath, error_desc)
        return
    datapath.set_version(max(usable_versions))
    self.logger.debug('move onto config mode')
    datapath.set_state(CONFIG_DISPATCHER)
    features_request = datapath.ofproto_parser.OFPFeaturesRequest(datapath)
    datapath.send_msg(features_request)