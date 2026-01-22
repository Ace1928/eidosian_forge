def check_field(self, field, **kwargs):
    errors = []
    if hasattr(self, 'check_field_type') and (not getattr(field, 'remote_field', None)):
        db_supports_all_required_features = all((getattr(self.connection.features, feature, False) for feature in field.model._meta.required_db_features))
        if db_supports_all_required_features:
            field_type = field.db_type(self.connection)
            if field_type is not None:
                errors.extend(self.check_field_type(field, field_type))
    return errors