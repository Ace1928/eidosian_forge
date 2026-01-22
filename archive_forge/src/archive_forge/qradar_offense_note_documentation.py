from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.ibm.qradar.plugins.module_utils.qradar import (

# FIXME - WOULD LIKE TO QUERY BY NAME BUT HOW TO ACCOMPLISH THAT IS NON-OBVIOUS
# offense_name:
#   description:
#    - Name of Offense
#   required: true
#   type: str

# FIXME - WOULD LIKE TO MANAGE STATE
# state:
#   description: Define state of the note: present or absent
#   required: false
#   choices: ["present", "absent"]
#   default: "present"
