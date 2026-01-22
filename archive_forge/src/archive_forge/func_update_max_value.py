def update_max_value(self):
    self.max_value = self.interval.upper()
    for child in self.children:
        if child:
            self.max_value = max(self.max_value, child.max_value)