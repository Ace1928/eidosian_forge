import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
def fake_get_fc_hbas_with_platform(self):
    return [{'ClassDevice': 'host1', 'ClassDevicePath': '/sys/devices/platform/smb/smb:motherboard/80040000000.peu0-c0/pci0000:00/0000:00:03.0/0000:05:00.2/host1/fc_host/host1', 'dev_loss_tmo': '30', 'fabric_name': '0x1000000533f55566', 'issue_lip': '<store method only>', 'max_npiv_vports': '255', 'maxframe_size': '2048 bytes', 'node_name': '0x200010604b019419', 'npiv_vports_inuse': '0', 'port_id': '0x680409', 'port_name': '0x100010604b019419', 'port_state': 'Online', 'port_type': 'NPort (fabric via point-to-point)', 'speed': '10 Gbit', 'supported_classes': 'Class 3', 'supported_speeds': '10 Gbit', 'symbolic_name': 'Emulex 554M FV4.0.493.0 DV8.3.27', 'tgtid_bind_type': 'wwpn (World Wide Port Name)', 'uevent': None, 'vport_create': '<store method only>', 'vport_delete': '<store method only>'}]