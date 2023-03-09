# other libs
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from scipy.stats import expon

# project 
from diagnosis import (
    import_ap_data, 
    plot_multiple_scenarios, 
    calc_bottleneck_frequency, 
    calc_bottleneck_severity)

# set defaults for plotting
plt.rcParams["font.family"] = "Times New Roman"


def plot_distribution(pts, scales, how="both", num_obs=10**6):
    """Helper function to display the effect of the exponential distribution on a given process time."""
    # create new figure and axis
    _cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(1, 1, figsize=(15*_cm, 10*_cm))
    # get x 
    x = np.linspace(
        start = 0, # expon.ppf(0.000001), 
        stop = 15, #expon.ppf(0.999999), 
        num = 100)
    # helper list for line styles
    lstyles = ["--", "-.", ":", "-"][:len(pts)]
    # iter process time and scales 
    for pt, sc, ls in zip(pts, scales, lstyles):
        # distribution as line plot 
        if how=="distr" or how =="both":
            # exponentially modified Gaussian distribution
            distr = expon(scale=pt*sc, loc=pt)
            # plot percent point function 
            ax.plot(
                x, 
                distr.pdf(x), 
                lw=2, 
                label=f'Exponential distribution (pt={pt})', 
                linestyle=ls, 
                alpha=0.85)
        # distribution as histogram 
        if how=="hist"or how=="both":
            # random variates of given type
            r = expon.rvs(size=num_obs, scale=pt*sc, loc=pt)
            ax.hist(r, 
                density=True, 
                bins="auto", 
                histtype='stepfilled', 
                alpha=0.2)

    # format axis
    ax.set_xlim(0, x[-1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Process time", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)

    # style 
    ax.legend(loc='best', frameon=True)
    ax.grid(color="silver")
    ax.set_title("Effect of exponential distribution on process times", fontsize=12)

    # save and show
    fig.savefig(f"images/distribution{pts}_{scales}.png", format="png", dpi=1200)
    fig.savefig(f"images/distribution{pts}_{scales}.svg", format="svg")
    plt.show()

def plot_distribution_comp(pts, scales, how="both", num_obs=10**6):
    """Helper function to display the effect of the exponential distribution on a given process time."""
    # create new figure and axis
    _cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(1, 2, figsize=(25*_cm, 7.5*_cm))
    # get x 
    x = np.linspace(
        start = 0, # expon.ppf(0.000001), 
        stop = 15, #expon.ppf(0.999999), 
        num = 100)
    # helper list for line styles
    lstyles = ["--", "-.", ":", "-"]
    dvari = {0.75 : "low", 1 : "neutral", 1.25: "high"}
    # iter process time and scales 
    for i, ax in enumerate(axes): 
        for pt, sc, ls in zip(pts[i], scales[i], lstyles):
            # distribution as line plot 
            if how=="distr" or how =="both":
                # exponentially modified Gaussian distribution
                distr = expon(scale=sc, loc=pt*sc)
                # plot percent point function 
                ax.plot(
                    x, 
                    distr.pdf(x), 
                    lw=2, 
                    label=f'pt={pt}, var={dvari[sc]}', 
                    linestyle=ls, 
                    alpha=0.85)
            # distribution as histogram 
            if how=="hist"or how=="both":
                # random variates of given type
                r = expon.rvs(size=num_obs, scale=pt*sc, loc=pt*sc)
                ax.hist(r, 
                    density=True, 
                    bins="auto", 
                    histtype='stepfilled', 
                    alpha=0.2)

            # format axis
            ax.set_xlim(0, x[-1])
            ax.set_ylim([0, 0.65])
            ax.set_xlabel("Process time", fontsize=12)
            ax.set_ylabel("Probability density", fontsize=12)

        # style 
        ax.legend(loc='best', frameon=True)
        ax.grid(color="silver")

    fig.suptitle("Effect of the exponential distribution and the added variability on the process times", y=0.95, fontsize=12)

    # save and show
    fig.savefig(f"images/distribution_comp_{pts}_{scales}.png", format="png", dpi=1200)
    fig.savefig(f"images/distribution_comp_{pts}_{scales}.svg", format="svg")


### plot distributions for the paper 

# distribution for S2 and S3
plot_distribution([2, 2.25], [1,1], how="distr")

# distribution for S1 
plot_distribution([2, 2, 2], [0.75, 1, 1.25], how="distr")

# both distributions comparative
plot_distribution_comp(
    pts=[[2, 2.25], [2, 2, 2]],
    scales=[[1, 1], [0.75, 1, 1.25]], 
    how="distr")


### plot (non-averaged!) rbs for a single simulation 

# clear previous figure 
plt.clf()

# import and prepare data 
df = import_ap_data("12k_S3-1", 0)
df = df.drop(labels="bottleneck", axis=1)
df = df.div(df.max(axis=1), axis=0)

# create new figure and axis
_cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(15*_cm, 10*_cm))

# format axis
ax.set_xlim(900, 1100)
ax.set_ylim(0, 1.15)
ax.set_xlabel("Time", loc="right", fontsize=12)
ax.set_ylabel("Relative bottleneck severity $\it{rbs}$", fontsize=12)

# custom line styles for black-and-white display
lstyles = [
    "solid", 
    "dotted", 
    "dashed", 
    "dashdot", 
    (0, (5,1)), # densely dashed
    (0, (3,1,1,1,1)), # densly dashdotted
    (0, (3,1,1,1,1,1))] # densely dashdotdotted

# fig 
fig.suptitle("Example of relative bottleneck severity for a single simulation (S3-1, run 0)", y=0.95, fontsize=12)
fig.tight_layout()

# iter over df to plot each row with custom style
for i, (col) in enumerate(df.columns): 
    if i in [0, 1, 2]: # skipped to maintain readable plot
        continue
    ax.plot(df[col][900:1100], label=col, linestyle=lstyles[i],)
    ax.legend(ncols=4,  handlelength=4, loc="upper center")

# set custom tick label for t_shifting 
tick_labels = [item.get_text() for item in ax.get_xticklabels()]
tick_labels[4] = 'Bottleneck\nshifting'
tick_values = [item for item in ax.get_xticks()]
tick_values[4] = 997.5
ax.set_xticklabels(tick_labels)
ax.set_xticks(tick_values)

# show or save
fig.savefig(fname=f"images/S3-1_example.png", format="png", dpi=1200)
fig.savefig(fname=f"images/S3-1_example.svg", format="svg")


### plot rbf-rbs comparison for one Scenario (3-1)

# clear previous figure 
plt.clf()

# S3 (two major bottlenecks)
layout = ["Relative bottleneck frequency", "Relative bottleneck severity"]
df_rbf = calc_bottleneck_frequency("12k_S3-1")
df_rbs = calc_bottleneck_severity("12k_S3-1")
df_comp = pd.DataFrame()
df_comp["S3-1_rbf"] = df_rbf["mean"]
df_comp["S3-1_rbs"] = df_rbs["mean"]


plot_multiple_scenarios(
    df = df_comp, 
    scenario = "12k_S3-1",
    plot_type = "comparison",
    plot_title = "Comparison of $\it{rbf}$ and $\it{rbs}$ for S3-1 (avg. of 10 runs)",
    layout = layout)

