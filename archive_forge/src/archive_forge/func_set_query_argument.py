def set_query_argument(self, name, key='*authmeth', value=None):
    """ choose authentication method based on a query argument """
    lookfor = '%s=%s' % (key, value or name)
    self.add_predicate(name, lambda environ: lookfor in environ.get('QUERY_STRING', ''))