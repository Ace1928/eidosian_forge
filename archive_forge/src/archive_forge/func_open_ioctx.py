def open_ioctx(self, *args, **kwargs):
    return mock_rados.ioctx()