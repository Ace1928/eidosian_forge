import time
import hashlib
import pyzor
def verify_signature(msg, user_key):
    """Verify that the provided message is correctly signed.

    The message must have "User", "Time", and "Sig" headers.

    If the signature is valid, then the function returns normally.
    If the signature is not valid, then a pyzor.SignatureError() exception
    is raised."""
    timestamp = int(msg['Time'])
    user = msg['User']
    provided_signature = msg['Sig']
    if abs(time.time() - timestamp) > pyzor.MAX_TIMESTAMP_DIFFERENCE:
        raise pyzor.SignatureError('Timestamp not within allowed range.')
    hashed_user_key = hash_key(user_key, user)
    del msg['Sig']
    correct_signature = sign_msg(hashed_user_key, timestamp, msg)
    if correct_signature != provided_signature:
        raise pyzor.SignatureError('Invalid signature.')