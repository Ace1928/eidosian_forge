import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def get_pbm_wsdl_location(vc_version):
    """Return PBM WSDL file location corresponding to VC version.

    :param vc_version: a dot-separated version string. For example, "1.2".
    :return: the pbm wsdl file location.
    """
    if not vc_version:
        return
    ver = vc_version.split('.')
    major_minor = ver[0]
    if len(ver) >= 2:
        major_minor = '%s.%s' % (major_minor, ver[1])
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    pbm_service_wsdl = os.path.join(curr_dir, 'wsdl', major_minor, 'pbmService.wsdl')
    if not os.path.exists(pbm_service_wsdl):
        LOG.warning('PBM WSDL file %s not found.', pbm_service_wsdl)
        return
    pbm_wsdl = urlparse.urljoin('file:', urllib.pathname2url(pbm_service_wsdl))
    LOG.debug('Using PBM WSDL location: %s.', pbm_wsdl)
    return pbm_wsdl