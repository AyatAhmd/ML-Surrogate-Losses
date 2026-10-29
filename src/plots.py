"""Notebook plots; figures are displayed inline and are not saved to folders."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'hinge': '#3266a8', 'logistic': '#d2752c', 'squared': '#31806b'}


def plot_learning_curves(result):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, hist in result.histories.items():
        color = COLORS[name]
        axes[0].plot(hist.epoch, hist.train_surrogate, label=name, color=color)
        axes[1].plot(hist.epoch, hist.train_error, label=f'{name} train', color=color)
        axes[1].plot(hist.epoch, hist.test_error, '--', label=f'{name} test', color=color)
    axes[0].set(title='Training surrogate risk', xlabel='Epoch', ylabel='Mean surrogate loss')
    axes[1].set(title='Classification error', xlabel='Epoch', ylabel='0-1 error')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(result.dataset.name.replace('_', ' ').title())
    fig.tight_layout()
    return fig


def plot_error_vs_surrogate(result):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (name, hist) in zip(axes, result.histories.items()):
        ax.plot(hist.train_surrogate, hist.train_error, label='Train', color=COLORS[name])
        ax.plot(hist.test_surrogate, hist.test_error, '--', label='Test', color=COLORS[name])
        ax.scatter(hist.train_surrogate.iloc[-1], hist.train_error.iloc[-1], color='black', s=22)
        ax.set(title=name.title(), xlabel='Mean surrogate loss')
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('0-1 error')
    axes[0].legend()
    fig.suptitle('Classification error vs surrogate loss (within each loss)')
    fig.tight_layout()
    return fig


def plot_decision_boundaries(result):
    if result.dataset.X.shape[1] != 2:
        raise ValueError('Only available for two-dimensional synthetic data')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True)
    x = result.dataset.X
    a = np.linspace(x[:, 0].min() - .4, x[:, 0].max() + .4, 140)
    b = np.linspace(x[:, 1].min() - .4, x[:, 1].max() + .4, 140)
    xx, yy = np.meshgrid(a, b)
    X_grid = result.scaler.transform(np.column_stack([xx.ravel(), yy.ravel()]))
    for ax, (name, model) in zip(axes, result.models.items()):
        zz = model.decision_function(X_grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-1e8, 0, 1e8], colors=['#f3d2ce','#d2e3f2'], alpha=.7)
        ax.contour(xx, yy, zz, levels=[0], colors=['black'], linewidths=1.2)
        ax.scatter(result.split.X_test[:,0], result.split.X_test[:,1],
                   c=result.split.y_test, cmap='coolwarm', s=12, alpha=.8)
        ax.set(title=name.title(), xlabel='x1')
    axes[0].set_ylabel('x2')
    fig.suptitle('Held-out observations and decision boundaries')
    fig.tight_layout()
    return fig


def plot_regularization(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, group in df.groupby('loss'):
        ordered = group.sort_values('l2')
        axes[0].plot(ordered.l2, ordered.test_error, '-o', label=name, color=COLORS[name])
        axes[1].plot(ordered.l2, ordered.weight_norm, '-o', label=name, color=COLORS[name])
    axes[0].set(ylabel='Test classification error', xlabel='L2 penalty')
    axes[1].set(ylabel='Weight norm', xlabel='L2 penalty')
    for ax in axes:
        ax.set_xscale('symlog', linthresh=.001)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Effect of regularization')
    fig.tight_layout()
    return fig
