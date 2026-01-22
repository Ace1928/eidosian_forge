from ironicclient.tests.functional.osc.v1 import base
Check baremetal chassis set and unset commands.

        Test steps:
        1) Create baremetal chassis in setUp.
        2) Set extra data for chassis.
        3) Check that baremetal chassis extra data was set.
        4) Unset extra data for chassis.
        5) Check that baremetal chassis extra data was unset.
        