import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
import torch

plt.rcParams["figure.figsize"] = (20,5)


def plot_seg(pred, label, x):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.plot(x, color='blue', linewidth=1, label='Original Signal')
    ax.plot(label, color='green', linewidth=3, label='Ground Truth')
    ax.plot(pred, color='red', linewidth=2, label='Prediction')
    ax.legend(fontsize=10)
    ax.set_ylim(-2, 2)

    canvas = FigureCanvas(fig)
    fig.set_tight_layout(True)

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def threshold_vector(vector, threshold=0.5):
    return (vector.cpu().numpy() > threshold).astype(int)


def compute_f1(y, y_pred, threshold=0.5):
    thresholded_logits = threshold_vector(y_pred, threshold)
    thresholded_y = threshold_vector(y, threshold)

    weighted_f1 = f1_score(thresholded_y, thresholded_logits, average='weighted', zero_division=1)
    samples_f1 = f1_score(thresholded_y, thresholded_logits, average='samples', zero_division=1)
    return torch.tensor(weighted_f1), torch.tensor(samples_f1)


def compute_iou(y, y_pred, threshold):
    thresholded_logits = threshold_vector(y_pred, threshold)
    thresholded_y = threshold_vector(y, threshold)

    weighted_iou = jaccard_score(thresholded_y, thresholded_logits, average='weighted')
    samples_iou = jaccard_score(thresholded_y, thresholded_logits, average='samples')
    return torch.tensor(weighted_iou), torch.tensor(samples_iou)
