import datetime
import time
from base64 import b64decode
from ansible.module_utils._text import to_bytes
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ec2_win_password(module):
    instance_id = module.params.get('instance_id')
    key_file = module.params.get('key_file')
    if module.params.get('key_passphrase') is None:
        b_key_passphrase = None
    else:
        b_key_passphrase = to_bytes(module.params.get('key_passphrase'), errors='surrogate_or_strict')
    if module.params.get('key_data') is None:
        b_key_data = None
    else:
        b_key_data = to_bytes(module.params.get('key_data'), errors='surrogate_or_strict')
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    client = module.client('ec2', retry_decorator=AWSRetry.jittered_backoff())
    if wait:
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=wait_timeout)
        while datetime.datetime.now() < end:
            data = _get_password(module, client, instance_id)
            decoded = b64decode(data)
            if not decoded:
                time.sleep(5)
            else:
                break
    else:
        data = _get_password(module, client, instance_id)
        decoded = b64decode(data)
    if wait and datetime.datetime.now() >= end:
        module.fail_json(msg=f'wait for password timeout after {int(wait_timeout)} seconds')
    if key_file is not None and b_key_data is None:
        try:
            with open(key_file, 'rb') as f:
                key = load_pem_private_key(f.read(), b_key_passphrase, default_backend())
        except IOError as e:
            module.fail_json(msg=f'I/O error ({int(e.errno)}) opening key file: {e.strerror}')
        except (ValueError, TypeError) as e:
            module.fail_json(msg='unable to parse key file')
    elif b_key_data is not None and key_file is None:
        try:
            key = load_pem_private_key(b_key_data, b_key_passphrase, default_backend())
        except (ValueError, TypeError) as e:
            module.fail_json(msg='unable to parse key data')
    try:
        decrypted = key.decrypt(decoded, PKCS1v15())
    except ValueError as e:
        decrypted = None
    if decrypted is None:
        module.fail_json(msg='unable to decrypt password', win_password='', changed=False)
    elif wait:
        elapsed = datetime.datetime.now() - start
        module.exit_json(win_password=decrypted, changed=False, elapsed=elapsed.seconds)
    else:
        module.exit_json(win_password=decrypted, changed=False)