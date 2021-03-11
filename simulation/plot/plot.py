import numpy as np
import matplotlib.pyplot as plt


def sir_plot(s, i, r, v=[]):
    fig, ax = plt.subplots(figsize=(10, 6))

    ticks = np.arange(0, s.shape[0], dtype=int)

    ax.plot(ticks, s, label="S", color="blue", marker=".")
    ax.plot(ticks, i, label="I", color="red", marker=".")
    ax.plot(ticks, r, label="R", color="green", marker=".")
    if len(v) > 0:
        ax.plot(ticks, v, label="V", color="turquoise", marker=".")

    ax.set_xticks(ticks)
    ax.set_xlabel('Week')
    ax.set_ylabel('Number of nodes')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.show()


def ni_plot(ni):
    fig, ax = plt.subplots(figsize=(10, 6))

    ticks = np.arange(0, ni.shape[0], dtype=int)

    ax.plot(ticks, ni, color="red", marker=".")

    ax.set_xticks(ticks)
    ax.set_xlabel('Week')
    ax.set_ylabel('Number of newly infected nodes')
    ax.grid(alpha=0.3)

    plt.show()


def ninv_plot(ni, nv):
    fig, ax = plt.subplots(figsize=(10, 6))

    ticks = np.arange(0, ni.shape[0], dtype=int)

    ax.plot(ticks, ni, color="red", label="newly I", marker=".")
    ax.plot(ticks, nv, color="turquoise", label="newly V", marker=".")

    ax.set_xticks(ticks)
    ax.set_xlabel('Week')
    ax.set_ylabel('Number of nodes')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.show()


def ni_comparison_plot(ni, ni_target):
    fig, ax = plt.subplots(figsize=(10, 6))

    ticks = np.arange(0, ni.shape[0], dtype=int)

    ax.plot(ticks, ni, color="red", label="newly I (simulation)", marker=".")
    ax.plot(ticks, ni_target, color="black", label="newly I (real)", linestyle="--")

    ax.set_xticks(ticks)
    ax.set_xlabel('Week')
    ax.set_ylabel('Number of nodes')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.show()
