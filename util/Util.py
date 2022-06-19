import matplotlib.pyplot as plt
import pandas as pd

def read_csv(path, sep=',', index_col=None, quoting=0, header='infer', verbose:bool = True):
    df = pd.read_csv(path, sep=sep, index_col=index_col, quoting=quoting, header=header)
    if verbose:
        print(f"* Reading CSV from path: {path}. Size: {df.shape}")
    return df


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