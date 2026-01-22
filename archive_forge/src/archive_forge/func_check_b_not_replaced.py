def check_b_not_replaced():
    test.assertEqual(b_uuid, next(iter(reality.resources_by_logical_name('B'))).uuid)
    test.assertIsNotNone(b_uuid)