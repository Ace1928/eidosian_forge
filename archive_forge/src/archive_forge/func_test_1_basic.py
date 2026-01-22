import unittest
import time
from boto.sdb.connection import SDBConnection
from boto.exception import SDBResponseError
def test_1_basic(self):
    print('--- running SDBConnection tests ---')
    c = SDBConnection()
    rs = c.get_all_domains()
    num_domains = len(rs)
    try:
        domain = c.create_domain('bad:domain:name')
    except SDBResponseError:
        pass
    domain_name = 'test%d' % int(time.time())
    domain = c.create_domain(domain_name)
    rs = c.get_all_domains()
    assert len(rs) == num_domains + 1
    item_1 = 'item1'
    same_value = 'same_value'
    attrs_1 = {'name1': same_value, 'name2': 'diff_value_1'}
    domain.put_attributes(item_1, attrs_1)
    item_2 = 'item2'
    attrs_2 = {'name1': same_value, 'name2': 'diff_value_2'}
    domain.put_attributes(item_2, attrs_2)
    item = domain.get_attributes(item_1, consistent_read=True)
    assert len(item.keys()) == len(attrs_1.keys())
    assert item['name1'] == attrs_1['name1']
    assert item['name2'] == attrs_1['name2']
    query = 'select * from %s where name1="%s"' % (domain_name, same_value)
    rs = domain.select(query, consistent_read=True)
    n = 0
    for item in rs:
        n += 1
    assert n == 2
    query = 'select * from %s where name2="diff_value_2"' % domain_name
    rs = domain.select(query, consistent_read=True)
    n = 0
    for item in rs:
        n += 1
    assert n == 1
    stat = domain.delete_attributes(item_1)
    assert stat
    item3 = {'name3_1': 'value3_1', 'name3_2': 'value3_2', 'name3_3': ['value3_3_1', 'value3_3_2']}
    item4 = {'name4_1': 'value4_1', 'name4_2': ['value4_2_1', 'value4_2_2'], 'name4_3': 'value4_3'}
    items = {'item3': item3, 'item4': item4}
    domain.batch_put_attributes(items)
    item = domain.get_attributes('item3', consistent_read=True)
    assert item['name3_2'] == 'value3_2'
    items = {'item3': item3}
    stat = domain.batch_delete_attributes(items)
    item = domain.get_attributes('item3', consistent_read=True)
    assert not item
    stat = domain.batch_delete_attributes({'item4': None})
    item = domain.get_attributes('item4', consistent_read=True)
    assert not item
    stat = c.delete_domain(domain)
    assert stat
    print('--- tests completed ---')