import os
def export_pdf(self):
    """Export to pdf."""
    if self.paginate:
        page_size = QPageSize(QPageSize.A4)
        page_layout = QPageLayout(page_size, QPageLayout.Portrait, QtCore.QMarginsF())
    else:
        factor = 0.75
        page_size = QPageSize(QtCore.QSizeF(self.size.width() * factor, self.size.height() * factor), QPageSize.Point)
        page_layout = QPageLayout(page_size, QPageLayout.Portrait, QtCore.QMarginsF())
    self.page().printToPdf(self.output_file, pageLayout=page_layout)