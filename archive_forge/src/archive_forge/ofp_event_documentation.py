import inspect
import time
from os_ken.controller import handler
from os_ken import ofproto
from . import event

    An event class to notify the port state changes of Dtatapath instance.

    This event performs like EventOFPPortStatus, but OSKen will
    send this event after updating ``ports`` dict of Datapath instances.
    An instance has at least the following attributes.

    ========= =================================================================
    Attribute Description
    ========= =================================================================
    datapath  os_ken.controller.controller.Datapath instance of the switch
    reason    one of OFPPR_*
    port_no   Port number which state was changed
    ========= =================================================================
    