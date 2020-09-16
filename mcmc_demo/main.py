from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

"""
Parameters
"""
initial_state = np.array([-20.0, 10.0]).transpose()

# Target distribution
mean = np.array([0.0, 0.0]).transpose()
cov = np.array([[5., 3.0],
                [3.0, 5.]])

# Proposal distribution (Gaussian)
scale = 2.0

# Visual
n_steps = 10000
trace_length = 300
plot_range = [-30, 30]
"""
Main Code
"""

fig = plt.figure(figsize=(20, 20))

def sample_from_proposal(point_prev):
    new_point = np.zeros_like(point_prev)
    new_point[0] = point_prev[0] + np.random.normal(scale=scale)
    new_point[1] = point_prev[1] + np.random.normal(scale=scale)
    return new_point


def proposal_pdf(point, point_prev):
    d = point - point_prev
    cov_prop = np.eye(2, 2)/scale
    tmp = np.transpose(d) @ cov_prop @ d
    p = (1.0 / np.sqrt(2 * np.pi * np.linalg.det(cov_prop))) * np.exp(-0.5 * tmp)
    return p


# Target pdf


def target_pdf(point):
    dx = point - mean
    tmp = np.transpose(dx) @ np.linalg.pinv(cov) @ dx
    p = (1.0 / np.sqrt(2 * np.pi * np.linalg.det(cov))) * np.exp(-0.5 * tmp)
    return p


def target_pdf_center():
    n_plot = 300
    x_ = np.linspace(plot_range[0], plot_range[1], n_plot)
    y_ = np.linspace(plot_range[0], plot_range[1], n_plot)
    x_pdf = []
    y_pdf = []
    for i in range(n_plot):
        p_x = target_pdf(np.array([x_[i], 0.]).transpose())
        x_pdf.append(p_x)

        p_y = target_pdf(np.array([0., y_[i]]).transpose())
        y_pdf.append(p_y)
    return x_, y_, x_pdf, y_pdf


x_, y_, x_pdf, y_pdf = target_pdf_center()

sample_trace_x = deque(maxlen=trace_length)
sample_trace_y = deque(maxlen=trace_length)
visit_x = deque()
visit_y = deque()
rejection_rage = deque()

if __name__ == '__main__':
    prev_sample = initial_state
    n_reject = 0
    for i in range(n_steps):
        # Sampling
        sample_proposed = sample_from_proposal(prev_sample)

        # Metropolis method
        r = target_pdf(sample_proposed)/target_pdf(prev_sample)
        alpha = min(1, r)
        if np.random.rand() < alpha:
            sample = sample_proposed
        else:
            #continue  # This continue is what I wanted to see how it works
            sample = prev_sample
            n_reject += 1
            print(f"Reject Rate @ {i+1}/{n_steps}: {n_reject/(i+1)}")

        # Save Sample
        sample_trace_x.append(sample[0])
        sample_trace_y.append(sample[1])
        visit_x.append(sample[0])
        visit_y.append(sample[1])
        prev_sample = sample

        # Visualization
        plt.clf()

        ax_y = plt.subplot(222)
        ax_y.plot(y_pdf, y_, "r")
        ax_y.hist(visit_y, orientation="horizontal", density=True)

        ax_x = plt.subplot(223)
        ax_x.plot(x_, x_pdf, "r")
        ax_x.hist(visit_x, density=True)

        ax_main = plt.subplot(221)
        # Error Ellipse
        evalue, evec = np.linalg.eig(cov)
        sq_eval = np.sqrt(evalue)
        for j in range(4):
            ell = Ellipse(
                xy=tuple(mean),
                width=evalue[0] * j * 2,
                height=evalue[1] * j * 2,
                angle=np.rad2deg(np.arccos(evec[0, 0]))
            )
            ell.set_facecolor("r")
            ell.set_alpha(0.1)
            ax_main.add_artist(ell)
        # Sample Trace
        ax_main.plot(sample_trace_x, sample_trace_y, "k.--", alpha=0.3)
        ax_main.set_xlim(plot_range)
        ax_main.set_ylim(plot_range)

        ax_rejection = plt.subplot(224)
        rejection_rage.append(n_reject/(i+1))
        ax_rejection.plot(rejection_rage, "k")
        ax_rejection.plot(np.zeros_like(rejection_rage) + 0.5, "b--")
        ax_rejection.set_ylim([-0.1, 1.1])
        ax_rejection.set_xlabel("step")
        ax_rejection.set_ylabel("rejection rate")

        plt.pause(0.01)

    print("Finish.")
    plt.show()
