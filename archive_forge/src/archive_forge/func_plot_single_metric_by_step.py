import json
import matplotlib.pyplot as plt
import argparse
import os
def plot_single_metric_by_step(data, metric_name, x_label, y_label, title, color):
    plt.plot(data[f'{metric_name}'], label=f'{title}', color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()