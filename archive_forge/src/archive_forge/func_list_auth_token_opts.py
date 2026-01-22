from keystonemiddleware import auth_token
def list_auth_token_opts():
    return auth_token.list_opts()