import logging
def register_table_methods(base_classes, **kwargs):
    base_classes.insert(0, TableResource)