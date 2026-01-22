import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
 execute tests.

        user: user specified value.
              eg. user = ('duration', (100, 100))
        on_wire: on-wire bytes
        header_bytes: header length
        