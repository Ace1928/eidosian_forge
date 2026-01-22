from gi.repository import GObject
import dbus.service
Initialize an exported GObject.

    :Parameters:
        `conn` : dbus.connection.Connection
            The D-Bus connection or bus
        `object_path` : str
            The object path at which to register this object.
    :Keywords:
        `bus_name` : dbus.service.BusName
            A bus name to be held on behalf of this object, or None.
        `gobject_properties` : dict
            GObject properties to be set on the constructed object.

            Any unrecognised keyword arguments will also be interpreted
            as GObject properties.
        