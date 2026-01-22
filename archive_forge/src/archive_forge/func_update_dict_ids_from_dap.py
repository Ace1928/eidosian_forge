from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@classmethod
def update_dict_ids_from_dap(cls, dct):
    if 'variablesReference' in dct:
        dct['variablesReference'] = cls._translate_id_from_dap(dct['variablesReference'])
    return dct