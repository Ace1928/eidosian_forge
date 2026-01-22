import json
import hashlib
import logging
from typing import List, Any, Dict
def detailed_process_data_security(data_security: Dict) -> str:
    encryption_info = data_security.get('Encryption', 'No Encryption Info')
    compliance_info = data_security.get('Compliance', 'No Compliance Info')
    return f'Encryption: {encryption_info}, Compliance: {compliance_info}'