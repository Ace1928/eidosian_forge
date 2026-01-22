from subprocess import PIPE
from subprocess import Popen
from saml2.extension.algsupport import DigestMethod
from saml2.extension.algsupport import SigningMethod
from saml2.sigver import get_xmlsec_binary
def get_algorithm_support(xmlsec):
    com_list = [xmlsec, '--list-transforms']
    pof = Popen(com_list, stderr=PIPE, stdout=PIPE)
    p_out, p_err = pof.communicate()
    p_out = p_out.decode('utf-8')
    p_err = p_err.decode('utf-8')
    if not p_err:
        p = p_out.splitlines()
        algs = [x.strip('"') for x in p[1].split(',')]
        digest = []
        signing = []
        for alg in algs:
            if alg in DIGEST_METHODS:
                digest.append(alg)
            elif alg in SIGNING_METHODS:
                signing.append(alg)
        return {'digest': digest, 'signing': signing}
    raise SystemError(p_err)