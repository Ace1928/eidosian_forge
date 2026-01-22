from xml import sax
from boto import handler
from boto.gs import acl
from tests.integration.gs.testcase import GSTestCase
def testVersionedBucketXmlAcl(self):
    b = self._MakeVersionedBucket()
    k = b.new_key('foo')
    s1 = 'test1'
    k.set_contents_from_string(s1)
    k = b.get_key('foo')
    g1 = k.generation
    s2 = 'test2'
    k.set_contents_from_string(s2)
    k = b.get_key('foo')
    g2 = k.generation
    acl1g1 = b.get_acl('foo', generation=g1)
    acl1g2 = b.get_acl('foo', generation=g2)
    owner1g1 = acl1g1.owner.id
    owner1g2 = acl1g2.owner.id
    self.assertEqual(owner1g1, owner1g2)
    entries1g1 = acl1g1.entries.entry_list
    entries1g2 = acl1g2.entries.entry_list
    self.assertEqual(len(entries1g1), len(entries1g2))
    acl_xml = '<ACCESSControlList><EntrIes><Entry>' + '<Scope type="AllUsers"></Scope><Permission>READ</Permission>' + '</Entry></EntrIes></ACCESSControlList>'
    aclo = acl.ACL()
    h = handler.XmlHandler(aclo, b)
    sax.parseString(acl_xml, h)
    b.set_acl(aclo, key_name='foo', generation=g1)
    acl2g1 = b.get_acl('foo', generation=g1)
    acl2g2 = b.get_acl('foo', generation=g2)
    entries2g1 = acl2g1.entries.entry_list
    entries2g2 = acl2g2.entries.entry_list
    self.assertEqual(len(entries2g2), len(entries1g2))
    public_read_entries1 = [e for e in entries2g1 if e.permission == 'READ' and e.scope.type == acl.ALL_USERS]
    public_read_entries2 = [e for e in entries2g2 if e.permission == 'READ' and e.scope.type == acl.ALL_USERS]
    self.assertEqual(len(public_read_entries1), 1)
    self.assertEqual(len(public_read_entries2), 0)