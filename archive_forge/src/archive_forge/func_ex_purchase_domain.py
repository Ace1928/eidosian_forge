from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def ex_purchase_domain(self, purchase_request):
    """
        Purchase a domain with GoDaddy

        :param  purchase_request: The completed document
            from ex_get_purchase_schema
        :type   purchase_request: ``dict``

        :rtype: :class:`GoDaddyDomainPurchaseResponse` Your order
        """
    result = self.connection.request('/v1/domains/purchase', data=purchase_request, method='POST').object
    return GoDaddyDomainPurchaseResponse(order_id=result['orderId'], item_count=result['itemCount'], total=result['total'], currency=result['currency'])