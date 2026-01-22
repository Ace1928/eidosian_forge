import time
def setup_chart(self):
    self.fig = bokeh.plotting.Figure(x_axis_type='datetime', x_axis_label='Training time')
    self.x_axis_val1 = []
    self.y_axis_val1 = []
    self.train1 = self.fig.line(self.x_axis_val1, self.y_axis_val1, line_dash='dotted', alpha=0.3, legend='train')
    self.train2 = self.fig.circle(self.x_axis_val1, self.y_axis_val1, size=1.5, line_alpha=0.3, fill_alpha=0.3, legend='train')
    self.train2.visible = False
    self.x_axis_val2 = []
    self.y_axis_val2 = []
    self.valid1 = self.fig.line(self.x_axis_val2, self.y_axis_val2, line_color='green', line_width=2, legend='validation')
    self.valid2 = self.fig.circle(self.x_axis_val2, self.y_axis_val2, line_color='green', line_width=2, legend=None)
    self.fig.legend.location = 'bottom_right'
    self.fig.yaxis.axis_label = self.metric_name
    return bokeh.plotting.show(self.fig, notebook_handle=True)