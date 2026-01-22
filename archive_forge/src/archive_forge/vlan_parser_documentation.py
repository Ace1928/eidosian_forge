from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError

    Input: Unsorted list of vlan integers
    Output: Sorted string list of integers according to IOS-like vlan list rules

    1. Vlans are listed in ascending order
    2. Runs of 3 or more consecutive vlans are listed with a dash
    3. The first line of the list can be first_line_len characters long
    4. Subsequent list lines can be other_line_len characters
    