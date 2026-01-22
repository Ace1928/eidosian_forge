import argparse
import mido
from mido import sockets
from mido.ports import MultiPort

Serve one or more output ports. Every message received on any of the
connected sockets will be sent to every output port.
