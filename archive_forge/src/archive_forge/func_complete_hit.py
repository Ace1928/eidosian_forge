from __future__ import absolute_import
from boto.mturk.test.support import unittest
def complete_hit(hit_type_id, response='Some Response'):
    verificationErrors = []
    sel = selenium(*sel_args)
    sel.start()
    sel.open('/mturk/welcome')
    sel.click('lnkWorkerSignin')
    sel.wait_for_page_to_load('30000')
    sel.type('email', 'boto.tester@example.com')
    sel.type('password', 'BotoTest')
    sel.click('Continue')
    sel.wait_for_page_to_load('30000')
    sel.open('/mturk/preview?groupId={hit_type_id}'.format(**vars()))
    sel.click('/accept')
    sel.wait_for_page_to_load('30000')
    sel.type('Answer_1_FreeText', response)
    sel.click('//div[5]/table/tbody/tr[2]/td[1]/input')
    sel.wait_for_page_to_load('30000')
    sel.click('link=Sign Out')
    sel.wait_for_page_to_load('30000')
    sel.stop()