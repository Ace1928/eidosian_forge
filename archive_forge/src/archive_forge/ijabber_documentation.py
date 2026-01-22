from zope.interface import Attribute, Interface

        Parent component has established a connection over the underlying
        transport.

        At this point, no traffic has been exchanged over the XML stream. This
        method can be used to change properties of the XML Stream (in C{xs}),
        the service manager or it's authenticator prior to stream
        initialization (including authentication).
        