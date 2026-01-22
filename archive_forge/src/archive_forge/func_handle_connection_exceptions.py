import logging
import re
from lxml import etree
from lxml.etree import QName
from ncclient.operations.retrieve import GetSchemaReply
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.juniper.rpc import GetConfiguration, LoadConfiguration, CompareConfiguration
from ncclient.operations.third_party.juniper.rpc import ExecuteRpc, Command, Reboot, Halt, Commit, Rollback
from ncclient.operations.rpc import RPCError
from ncclient.xml_ import to_ele, replace_namespace, BASE_NS_1_0, NETCONF_MONITORING_NS
from ncclient.transport.third_party.junos.parser import JunosXMLParser
from ncclient.transport.parser import DefaultXMLParser
from ncclient.transport.parser import SAXParserHandler
def handle_connection_exceptions(self, sshsession):
    c = sshsession._channel = sshsession._transport.open_channel(kind='session')
    c.set_name('netconf-command-' + str(sshsession._channel_id))
    c.exec_command('xml-mode netconf need-trailer')
    return True