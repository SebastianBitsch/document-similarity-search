from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd

def read_csv(path, sep=',', index_col=None, quoting=0, header='infer', verbose:bool = True):
    df = pd.read_csv(path, sep=sep, index_col=index_col, quoting=quoting, header=header)
    if verbose:
        print(f"* Reading CSV from path: {path}. Size: {df.shape}")
    return df


def plot_precision_recall(y_test, y_preds_proba, title:str):
    #calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_test, y_preds_proba)

    #create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision)

    #add axis labels to plot
    ax.set_title(title)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    return fig, ax

def plot_accuracies(accuracies: dict, title: str):
    """
    Plot a dictionary of accuracies on the format {"Computer vision": 0.99, (...) }
    """
    
    f, ax = plt.subplots(figsize=(18,4))

    plt.bar(range(len(accuracies)), list(accuracies.values()), align='center')
    plt.xticks(range(len(accuracies)), list(accuracies.keys()), rotation=90)
    plt.ylim([0,1.1])

    ax.set_title(title)
    ax.yaxis.grid()
    ax.set_axisbelow(True)


    for bars in ax.containers:
        ax.bar_label(bars)
    return f, ax