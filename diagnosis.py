import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def import_data(scenario: str, number: int,) -> pd.DataFrame:
    # import csv from results "data/"
    df = pd.read_csv(f"data/active_periods_{scenario}_{number}.csv", index_col=0)
    # transform bottleneck column to int
    df["bottleneck"] = [int(b[1:]) for b in df["bottleneck"]]
    # remove the first 2.000 observations to get 10.080 rows
    df = df[1999:]
    return df.reset_index(drop=True)


def calc_bottleneck_frequency(scenario: str):
    # get result df
    bn_frequencies = pd.DataFrame()
    # plot for bottleneck frequencies
    for i in range(10):
        # import data
        df = import_data(scenario, i)

        # get normalized and sorted count of all bottleneck stations
        frequencies = (
            df["bottleneck"]
            .value_counts(normalize=True, sort=False, ascending=False, dropna=False)
            .sort_index()
        )
        
        # create new Series for length of df (needed to handle zero)
        bn_freq = pd.Series([0] * 7)
        # iter over frequencies and update
        for idx, val in frequencies.items():
            bn_freq[idx] = val

        # add col to result df
        bn_frequencies[str(i)] = bn_freq.values

    # calc some metrics for plotting
    bn_frequencies["mean"] = (
        bn_frequencies[bn_frequencies.columns[:10]].mean(axis=1).values
    )
    bn_frequencies["var"] = (
        bn_frequencies[bn_frequencies.columns[:10]].var(axis=1).values
    )
    bn_frequencies["std"] = (
        bn_frequencies[bn_frequencies.columns[:10]].std(axis=1).values
    )
    bn_frequencies["min"] = (
        bn_frequencies[bn_frequencies.columns[:10]].min(axis=1).values
    )
    bn_frequencies["max"] = (
        bn_frequencies[bn_frequencies.columns[:10]].max(axis=1).values
    )
    # return df
    return bn_frequencies


def plot_bottleneck_frequency(
    df: pd.DataFrame, plot_title: str, plot_min_max: bool = True, plot_std: bool = True,
) -> None:
    # some plot defaults
    color1 = "silver"
    color2 = "lightgray"
    color3 = "gray"
    color4 = "black"
    font_size = 14

    # create new figure and axis
    fig, ax = plt.subplots()

    # plot min_max
    if plot_min_max:
        ax.fill_between(x=range(7), y1=df["max"], y2=df["min"], color=color2, alpha=0.5)

    # plot standard deviation
    if plot_std:
        var_upper = df["mean"] + df["std"]
        var_lower = df["mean"] - df["std"]
        ax.fill_between(x=range(7), y1=var_upper, y2=var_lower, color=color3, alpha=0.5)

    # plot average values
    ax.plot(df["mean"], color=color4)

    # ax limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 6)
    # ax ticks
    xtick_labels = df.index.tolist()
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    # ax labels
    ax.set_ylabel("Relative bottleneck frequency [%]")
    ax.set_xlabel("Station number")
    # add grid
    ax.grid(color=color1)
    # add legend
    custom_legend = [
        Line2D([0], [0], color=color4, lw=4),
        Line2D([0], [0], color=color3, lw=4),
        Line2D([0], [0], color=color2, lw=4),
    ]
    ax.legend(
        custom_legend,
        ["Avg. bottleneck frequency", "Span of Min. and Max.", "Span of Standard Dev."],
        loc="upper center",
    )
    # add title
    ax.set_title(f"{plot_title[4:]}: Avg. rel. bottleneck frequency (of 10 sim. runs)")

    # show or save
    fig.savefig(fname=f"images/{plot_title}.png", dpi=300, format="png")
    # plt.show()


def plot_multiple_scenarios(df: pd.DataFrame, name: str, layout: list[str]):
    # some plot defaults
    line_styles = ["-.", ":", "--"]

    # create new figure and axis
    fig, ax = plt.subplots()

    # iter over scenarios
    for col in range(len(df.columns)):
        ax.plot(
            df[df.columns[col]],
            label=f"{name[4:]}{col+1}: {layout[col]}",
            linestyle=line_styles[col],
        )

    # ax limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 6)
    # ax ticks
    xtick_labels = df.index.tolist()
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    # ax labels
    ax.set_ylabel("Relative bottleneck frequency [%]")
    ax.set_xlabel("Station number")
    # add grid
    ax.grid(color="silver")
    # add legend
    ax.legend()
    # add title
    ax.set_title(f"{scenario[4:-2]}: Avg. rel. bottleneck frequency (of 10 sim. runs)")

    # show or save
    # plt.show()
    fig.savefig(fname=f"images/{name[:-1]}.png", dpi=300, format="png")


# plot every scenario individually
for s in range(1, 4):
    for i in range(1, 4):
        scenario = f"12k_S{s}-{i}"
        df = calc_bottleneck_frequency(scenario)
        plot_bottleneck_frequency(df, scenario)

# plot every scenario together

# S1
layout_s1 = [
    "□–□–□–□–□–□–□ (low var.)",
    "□–□–□–□–□–□–□ (med. var)",
    "□–□–□–□–□–□–□ (high var.)",
]
df_scen = pd.DataFrame()
for i in range(1, 4):
    scenario = f"12k_S1-{i}"
    df = calc_bottleneck_frequency(scenario)
    df_scen[scenario] = df["mean"]
plot_multiple_scenarios(df_scen, scenario[:-1], layout_s1)

# S2
layout_s2 = ["□–■–□–□–□–□–□", "□–□–□–■–□–□–□", "□–□–□–□–□–■–□"]
df_scen = pd.DataFrame()
for i in range(1, 4):
    scenario = f"12k_S2-{i}"
    df = calc_bottleneck_frequency(scenario)
    df_scen[scenario] = df["mean"]
plot_multiple_scenarios(df_scen, scenario[:-1], layout_s2)

# S3
layout_s3 = ["□–□–■–□–■–□–□", "□–■–□–□–□–■–□", "■–□–□–□–□–□–■"]
df_scen = pd.DataFrame()
for i in range(1, 4):
    scenario = f"12k_S3-{i}"
    df = calc_bottleneck_frequency(scenario)
    df_scen[scenario] = df["mean"]
plot_multiple_scenarios(df_scen, scenario[:-1], layout_s3)


'''# plot for bottleneck frequencies
for i in range(10):
    # get data
    df = import_data("12k_S1-1", i)
    # drop bottleneck col
    df = df.drop(labels="bottleneck", axis=1)
    # normalize values for each row according to max in row
    df = df.div(df.max(axis=1), axis=0)
    bn_seve = df.mean()
    plt.plot(bn_seve, label=i)
plt.legend()

'''