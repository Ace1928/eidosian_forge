import json
def ebay_compliance_fix(session):

    def _compliance_fix(response):
        token = json.loads(response.text)
        if token.get('token_type') in ['Application Access Token', 'User Access Token']:
            token['token_type'] = 'Bearer'
            fixed_token = json.dumps(token)
            response._content = fixed_token.encode()
        return response
    session.register_compliance_hook('access_token_response', _compliance_fix)
    session.register_compliance_hook('refresh_token_response', _compliance_fix)
    return session