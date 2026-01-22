from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import sample_router

sample router manager.
(un-)instantiate routers
Usage example:
osken-manager --verbose     os_ken.services.protocols.vrrp.manager     os_ken.services.protocols.vrrp.dumper     os_ken.services.protocols.vrrp.sample_manager
