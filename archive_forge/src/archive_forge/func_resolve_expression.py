from django.db.models.expressions import OrderByList
def resolve_expression(self, *args, **kwargs):
    self.order_by = self.order_by.resolve_expression(*args, **kwargs)
    return super().resolve_expression(*args, **kwargs)