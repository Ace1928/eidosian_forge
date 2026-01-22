import base64
import logging
import netaddr
from os_ken.lib import dpid
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_2
def send_experimenter(dp, exp, logger=None):
    experimenter = exp.get('experimenter', 0)
    exp_type = exp.get('exp_type', 0)
    data_type = exp.get('data_type', 'ascii')
    data = exp.get('data', '')
    if data_type == 'base64':
        data = base64.b64decode(data)
    elif data_type == 'ascii':
        data = data.encode('ascii')
    else:
        get_logger(logger).error('Unknown data type: %s', data_type)
        return
    expmsg = dp.ofproto_parser.OFPExperimenter(dp, experimenter, exp_type, data)
    send_msg(dp, expmsg, logger)