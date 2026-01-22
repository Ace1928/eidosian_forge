import os
def is_prod_appengine():
    return 'APPENGINE_RUNTIME' in os.environ and os.environ.get('SERVER_SOFTWARE', '').startswith('Google App Engine/')