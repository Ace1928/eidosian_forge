def writeToken(self, token, data):
    if token <= 255 and token >= 0:
        data.append(token)
    else:
        raise ValueError('Invalid token: %s' % token)