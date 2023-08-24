import base64
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns


def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode("utf-8")
    buffer.close()
    return graph


def get_sns_plot(ah):
    plt.switch_backend("AGG")
    plt.figure(figsize=(3, 5))
    sns.countplot(ah)

    plt.title("Count Plot OF Dataset Extracted")

    graph = get_graph()
    return graph
