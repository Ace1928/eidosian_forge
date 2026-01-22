from django.db.models.expressions import OrderByList
def get_source_expressions(self):
    if self.order_by.source_expressions:
        return super().get_source_expressions() + [self.order_by]
    return super().get_source_expressions()