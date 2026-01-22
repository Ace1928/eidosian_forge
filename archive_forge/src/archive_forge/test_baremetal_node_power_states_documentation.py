from ironicclient.tests.functional.osc.v1 import base
Reboot node from Power ON state.

        Test steps:
        1) Create baremetal node in setUp.
        2) Set node Power State ON as precondition.
        3) Call reboot command for baremetal node.
        4) Check node Power State ON in node properties.
        