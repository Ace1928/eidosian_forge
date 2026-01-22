from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
@staticmethod
def call_parameters():
    return '\n        {\n            "location": {\n                "value": "string"\n            },\n            "virtualMachineName": {\n                "value": "string"\n            },\n            "virtualMachineSize": {\n                "value": "string"\n            },\n            "networkSecurityGroupName": {\n                "value": "string"\n            },\n            "adminUsername": {\n                "value": "string"\n            },\n            "virtualNetworkId": {\n                "value": "string"\n            },\n            "adminPassword": {\n                "value": "string"\n            },\n            "subnetId": {\n                "value": "string"\n            },\n            "customData": {\n                "value": "string"\n            },\n            "environment": {\n                "value": "prod"\n            },\n            "storageAccount": {\n                "value": "string"\n            }\n        }\n        '