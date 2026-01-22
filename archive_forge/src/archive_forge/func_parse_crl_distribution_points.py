from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def parse_crl_distribution_points(module, crl_distribution_points):
    result = []
    for index, parse_crl_distribution_point in enumerate(crl_distribution_points):
        try:
            params = dict(full_name=None, relative_name=None, crl_issuer=None, reasons=None)
            if parse_crl_distribution_point['full_name'] is not None:
                if not parse_crl_distribution_point['full_name']:
                    raise OpenSSLObjectError('full_name must not be empty')
                params['full_name'] = [cryptography_get_name(name, 'full name') for name in parse_crl_distribution_point['full_name']]
            if parse_crl_distribution_point['relative_name'] is not None:
                if not parse_crl_distribution_point['relative_name']:
                    raise OpenSSLObjectError('relative_name must not be empty')
                try:
                    params['relative_name'] = cryptography_parse_relative_distinguished_name(parse_crl_distribution_point['relative_name'])
                except Exception:
                    if CRYPTOGRAPHY_VERSION < LooseVersion('1.6'):
                        raise OpenSSLObjectError('Cannot specify relative_name for cryptography < 1.6')
                    raise
            if parse_crl_distribution_point['crl_issuer'] is not None:
                if not parse_crl_distribution_point['crl_issuer']:
                    raise OpenSSLObjectError('crl_issuer must not be empty')
                params['crl_issuer'] = [cryptography_get_name(name, 'CRL issuer') for name in parse_crl_distribution_point['crl_issuer']]
            if parse_crl_distribution_point['reasons'] is not None:
                reasons = []
                for reason in parse_crl_distribution_point['reasons']:
                    reasons.append(REVOCATION_REASON_MAP[reason])
                params['reasons'] = frozenset(reasons)
            result.append(cryptography.x509.DistributionPoint(**params))
        except (OpenSSLObjectError, ValueError) as e:
            raise OpenSSLObjectError('Error while parsing CRL distribution point #{index}: {error}'.format(index=index, error=e))
    return result