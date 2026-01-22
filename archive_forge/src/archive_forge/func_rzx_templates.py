from enum import Enum
from typing import List, Dict
from qiskit.circuit.library.templates import rzx
def rzx_templates(template_list: List[str]=None) -> Dict:
    """Convenience function to get the cost_dict and templates for template matching.

    Args:
        template_list: List of instruction names.

    Returns:
        Decomposition templates and cost values.
    """
    if template_list is None:
        template_list = ['zz1', 'zz2', 'zz3', 'yz', 'xz', 'cy']
    templates = [RZXTemplateMap[gate.upper()].value for gate in template_list]
    cost_dict = {'rzx': 0, 'cx': 6, 'rz': 0, 'sx': 1, 'p': 0, 'h': 1, 'rx': 1, 'ry': 1}
    rzx_dict = {'template_list': templates, 'user_cost_dict': cost_dict}
    return rzx_dict