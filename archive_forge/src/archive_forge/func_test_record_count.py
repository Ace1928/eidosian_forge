from tests.compat import unittest
from tests.integration.route53 import Route53TestCase
from boto.route53.record import ResourceRecordSets
def test_record_count(self):
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    hosts = 101
    for hostid in range(hosts):
        rec = 'test' + str(hostid) + '.%s' % self.base_domain
        created = rrs.add_change('CREATE', rec, 'A')
        ip = '192.168.0.' + str(hostid)
        created.add_value(ip)
        if (hostid + 1) % 100 == 0:
            rrs.commit()
            rrs = ResourceRecordSets(self.conn, self.zone.id)
    rrs.commit()
    all_records = self.conn.get_all_rrsets(self.zone.id)
    i = 0
    for rset in all_records:
        i += 1
    i = 0
    for rset in all_records:
        i += 1
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    for hostid in range(hosts):
        rec = 'test' + str(hostid) + '.%s' % self.base_domain
        deleted = rrs.add_change('DELETE', rec, 'A')
        ip = '192.168.0.' + str(hostid)
        deleted.add_value(ip)
        if (hostid + 1) % 100 == 0:
            rrs.commit()
            rrs = ResourceRecordSets(self.conn, self.zone.id)
    rrs.commit()
    records = hosts + 2
    self.assertEqual(i, records)