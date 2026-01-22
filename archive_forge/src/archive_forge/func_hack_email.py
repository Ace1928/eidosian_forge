def hack_email():
    """The python2.6 version of email.message_from_string, doesn't work with
    unicode strings. And in python3 it will only work with a decoded.

    So switch to using only message_from_bytes.
    """
    import email
    if not hasattr(email, 'message_from_bytes'):
        email.message_from_bytes = email.message_from_string