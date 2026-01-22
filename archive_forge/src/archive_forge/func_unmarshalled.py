from suds import *
from logging import getLogger
def unmarshalled(self, context):
    """
        Suds has unmarshalled the received reply.

        Provides the plugin with the opportunity to inspect/modify the
        unmarshalled reply object before it is returned.

        @param context: The reply context.
            The I{reply} is unmarshalled suds object.
        @type context: L{MessageContext}

        """
    pass