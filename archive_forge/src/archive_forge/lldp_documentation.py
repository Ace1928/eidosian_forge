import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
Organizationally Specific TLV encoder/decoder class

    ================= =============================================
    Attribute         Description
    ================= =============================================
    buf               Binary data to parse.
    oui               Organizationally unique ID.
    subtype           Organizationally defined subtype.
    info              Organizationally defined information string.
    ================= =============================================
    