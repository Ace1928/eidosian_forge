import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
def unify_credentials(username: typing.Optional[typing.Union[str, Credential, typing.List[Credential]]]=None, password: typing.Optional[str]=None, required_protocol: typing.Optional[str]=None) -> typing.List[Credential]:
    """Process user input credentials.

    Converts the user facing credential input into a list of credentials that
    is known by spnego. Also filters out any duplicate credentials/ones that
    are rendered obsolete by any ones preceding it. For example NTLMHash won't
    be used if it's specified after Password in the credential list.

    Args:
        username: The username or list of credentials to process.
        password: The password for username when it's a string.
        required_protocol: Optionally checks that at least 1 credential must
            support the protocol specified.

    Returns:
        typing.List[Credential]: A list of credentials based on the free-form
        input.
    """
    if username:
        if isinstance(username, str):
            if not password:
                username = [CredentialCache(username=username)]
            elif is_ntlm_hash(password):
                lm, nt = password.split(':', 1)
                username = [NTLMHash(username=username, lm_hash=lm, nt_hash=nt)]
            else:
                username = [Password(username=username, password=password)]
        elif not isinstance(username, list):
            username = [username]
    else:
        username = [CredentialCache()]
    credentials: typing.List[Credential] = []
    used_protocols: typing.Set[str] = set()
    for cred in username:
        if not isinstance(cred, (CredentialCache, KerberosCCache, KerberosKeytab, NTLMHash, Password)):
            raise InvalidCredentialError(context_msg='Invalid username/credential specified, must be a string or Credential object.')
        cred_useful = False
        for cred_protocol in cred.supported_protocols:
            if cred_protocol not in used_protocols:
                used_protocols.add(cred_protocol)
                cred_useful = True
        if cred_useful:
            credentials.append(cred)
    if required_protocol and required_protocol not in used_protocols:
        found_protocols = ', '.join(sorted(used_protocols))
        raise NoCredentialError(context_msg=f'A credential for {required_protocol} is needed but only found credentials for {found_protocols}')
    return credentials