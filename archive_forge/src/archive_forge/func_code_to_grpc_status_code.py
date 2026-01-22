import grpc
def code_to_grpc_status_code(code):
    try:
        return _CODE_TO_GRPC_CODE_MAPPING[code]
    except KeyError:
        raise ValueError('Invalid status code %s' % code)