import logging
import sys
from oslo_messaging import exceptions
from oslo_messaging.rpc import dispatcher as rpc_dispatcher
from oslo_messaging import server as msg_server
from oslo_messaging import transport as msg_transport
def expose(func):
    """Decorator for RPC endpoint methods that are exposed to the RPC client.

    If the dispatcher's access_policy is set to ExplicitRPCAccessPolicy then
    endpoint methods need to be explicitly exposed.::

        # foo() cannot be invoked by an RPC client
        def foo(self):
            pass

        # bar() can be invoked by an RPC client
        @rpc.expose
        def bar(self):
            pass

    """
    func.exposed = True
    return func