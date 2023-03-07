# other
import pandas as pd
from matplotlib import lines
from matplotlib import pyplot as plt


def import_ap_data(scenario: str, number: int,) -> pd.DataFrame:
    # import csv from results "data/"
    df = pd.read_csv(f"data/active_periods_{scenario}_{number}.csv", index_col=0)
    # transform bottleneck column to int
    df["bottleneck"] = [int(b[1:]) for b in df["bottleneck"]]
    # remove the first 2.000 observations to get 10.080 rows
    df = df[1999:]
    return df.reset_index(drop=True)


def get_metrics(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns input df with additional cols with mean, var, stdev, min, max'''
    df["mean"] = (df[df.columns[:10]].mean(axis=1).values)
    df["var"] = (df[df.columns[:10]].var(axis=1).values)
    df["std"] = (df[df.columns[:10]].std(axis=1).values)
    df["min"] = (df[df.columns[:10]].min(axis=1).values)
    df["max"] = (df[df.columns[:10]].max(axis=1).values)
    return df

def get_plot_type(t: str) -> str:
    # get plot type for some labeling
    if t=="rbf":
        return "Relative bottleneck frequency"
    elif t=="rbs":
        return "Relative bottleneck severity"
    elif t=="comparison":
        return "Relative bottleneck metric (rbf and rbs)"
    else:
        raise ValueError(f"Invalid lable type input: {t}")
    

def calc_bottleneck_frequency(scenario_name: str):
    '''Get relative bottleneck frequency (rbf)'''
    # get result df
    bn_frequencies = pd.DataFrame()
    # plot for bottleneck frequencies
    for i in range(10):
        # import data
        df = import_ap_data(scenario_name, i)
        # get normalized and sorted count of all bottleneck stations
        frequencies = (df["bottleneck"]
            .value_counts(normalize=True, sort=False, ascending=False, dropna=False).sort_index())
        # create new Series for length of df (needed to handle zero)
        bn_freq = pd.Series([0] * 7)
        # iter over frequencies and update
        for idx, val in frequencies.items():
            bn_freq[idx] = val
        # add col to result df
        bn_frequencies[str(i)] = bn_freq.values
    # calc some metrics for plotting
    bn_frequencies = get_metrics(bn_frequencies)
    # return df
    return bn_frequencies


def calc_bottleneck_severity(scenario_name: str):
    # get result df
    bn_severity = pd.DataFrame()
    # plot for bottleneck severities
    for i in range(10):
        # import data
        df = import_ap_data(scenario_name, i)
        # drop bottleneck col
        df = df.drop(labels="bottleneck", axis=1)
        # normalize values for each row according to max in row
        df = df.div(df.max(axis=1), axis=0)
        # calc mean severty for each station 
        bn_seve = df.mean()
        # add senario to result df 
        bn_severity[str(i)] = bn_seve.values
    # calc some metrics for plotting
    bn_severity = get_metrics(bn_severity)
    # return df
    return bn_severity


def plot_avg_scenarios(
    df: pd.DataFrame, 
    scenario: str,
    plot_title: str, 
    plot_type: str,
    plot_min_max: bool = True, 
    plot_std: bool = True,
    ) -> None:
    
    # some plot defaults
    color1 = "silver"
    color2 = "lightgray"
    color3 = "gray"
    color4 = "black"
    font_size = 12

    # create new figure and axis
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(15*cm, 10*cm))

    # plot min_max
    if plot_min_max:
        ax.fill_between(x=range(7), y1=df["max"], y2=df["min"], color=color2, alpha=0.5)

    # plot standard deviation
    if plot_std:
        var_upper = df["mean"] + df["std"]
        var_lower = df["mean"] - df["std"]
        ax.fill_between(x=range(7), y1=var_upper, y2=var_lower, color=color3, alpha=0.5)

    # get plot type
    p_type = get_plot_type(plot_type)

    # plot average values
    ax.plot(df["mean"], color=color4)

    # ax limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 6)
    # ax ticks
    xtick_labels = [f"S{i}" for i in df.index.tolist()]
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    # ax labels
    ax.set_ylabel(f"{p_type}", fontsize=font_size)
    ax.set_xlabel("Station", fontsize=font_size)
    # add grid
    ax.grid(color=color1)
    # add legend
    custom_legend = [
        lines.Line2D([0], [0], color=color4, lw=2),
        lines.Line2D([0], [0], color=color3, lw=5),
        lines.Line2D([0], [0], color=color2, lw=5),
    ]
    ax.legend(
        custom_legend,
        [p_type, 
         "Span of Min. and Max.", 
         "Span of Standard Dev."],
        loc="upper center",
        fontsize=font_size)
    # add title
    ax.set_title(plot_title, fontsize=font_size)
    # save
    fig.savefig(fname=f"images/{plot_type}/{scenario}_{plot_type}.png", format="png", dpi=600)
    fig.savefig(fname=f"images/{plot_type}/{scenario}_{plot_type}.svg", format="svg")


def plot_multiple_scenarios(
    df: pd.DataFrame, 
    scenario: str, 
    plot_type: str,
    plot_title: str,
    layout: list[str]):

    # some plot defaults
    font_size = 12
    line_styles = ["-.", ":", "--", "-.", ":", "--"]

    # get plot type
    p_type = get_plot_type(plot_type)

    # create new figure and axis
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(15*cm, 10*cm))

    # iter over scenarios
    for col in range(len(df.columns)):
        ax.plot(
            df[df.columns[col]],
            label=f"{scenario[4:]}{col+1}: {layout[col]}",
            linestyle=line_styles[col])

    # ax limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 6)
    # ax ticks
    xtick_labels = [f"S{i}" for i in df.index.tolist()]
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    # ax labels
    ax.set_ylabel(p_type, fontsize=font_size)
    ax.set_xlabel("Station", fontsize=font_size)
    # add grid
    ax.grid(color="silver")
    # add legend
    ax.legend(loc="upper right")
    # add title
    ax.set_title(plot_title, fontsize=font_size)

    # show or save
    fig.savefig(fname=f"images/{scenario}_{plot_type}.png", format="png", dpi=1200)
    fig.savefig(fname=f"images/{scenario}_{plot_type}.svg", format="svg")


if __name__ == "__main__":
        
    ### rbf: relative bottleneck FREQUENCY (every scenario individually) ###
    for s in range(1, 4): # S1, S2, S3
        for i in range(1, 4): # -1, -2, -3
            # calculate rbf
            scenario = f"12k_S{s}-{i}"
            df = calc_bottleneck_frequency(scenario_name = scenario)
            plot_title = f"Relative bottleneck frequency for {scenario[4:]} (avg. of 10 runs)"
            # plot rbf
            plot_avg_scenarios(
                df = df, 
                scenario = scenario,
                plot_title = plot_title, 
                plot_type = "rbf")


    ### rbs: relative bottleneck SEVERITY (every scenario individually) ###
    for s in range(1, 4): # S1, S2, S3
        for i in range(1, 4): # -1, -2, -3
            # calculate rbs
            scenario = f"12k_S{s}-{i}"
            df = calc_bottleneck_frequency(scenario_name = scenario)
            plot_title = f"Relative bottleneck severity for {scenario[4:]} (avg. of 10 runs)"
            # plot rbs
            plot_avg_scenarios(
                df = df, 
                scenario = scenario,
                plot_title = plot_title, 
                plot_type = "rbs")

    ### rbf: relative bottleneck FREQUENCY (every scenario individually) ###

    for s in range(1, 4): # S1, S2, S3
        for i in range(1, 4): # -1, -2, -3
            # calculate rbf
            scenario = f"12k_S{s}-{i}"
            df = calc_bottleneck_frequency(scenario_name = scenario)
            plot_title = f"Relative bottleneck frequency for {scenario[4:]} (avg. of 10 runs)"
            # plot rbf
            plot_avg_scenarios(
                df = df, 
                scenario = scenario,
                plot_title = plot_title, 
                plot_type = "rbf")


    ### rbs: relative bottleneck SEVERITY (every scenario individually) ###
    for s in range(1, 4): # S1, S2, S3
        for i in range(1, 4): # -1, -2, -3
            # calculate rbs
            scenario = f"12k_S{s}-{i}"
            df = calc_bottleneck_frequency(scenario_name = scenario)
            plot_title = f"Relative bottleneck severity for {scenario[4:]} (avg. of 10 runs)"
            # plot rbs
            plot_avg_scenarios(
                df = df, 
                scenario = scenario,
                plot_title = plot_title, 
                plot_type = "rbs")


    ### Relative botttleneck frequency (comparison) ###

    # S1 (change in variabilities)
    layout_s1 = ["□–□–□–□–□–□–□ (low var.)",
                "□–□–□–□–□–□–□ (med. var)",
                "□–□–□–□–□–□–□ (high var.)"]
    df_s1 = pd.DataFrame()
    for i in range(1, 4):
        s1 = f"12k_S1-{i}"
        df = calc_bottleneck_frequency(s1)
        df_s1[s1] = df["mean"]
    plot_multiple_scenarios(
        df = df_s1, 
        scenario = s1,
        plot_type = "rbf",
        plot_title = f"Relative bottleneck frequency for {s1[4:6]} (avg. of 10 runs)",
        layout = layout_s1)

    # S2 (one major bottleneck)
    layout_s2 = ["□–■–□–□–□–□–□", 
                "□–□–□–■–□–□–□", 
                "□–□–□–□–□–■–□"]
    df_s2 = pd.DataFrame()
    for i in range(1, 4):
        s2 = f"12k_S2-{i}"
        df = calc_bottleneck_frequency(s2)
        df_s2[s2] = df["mean"]
    plot_multiple_scenarios(
        df = df_s2, 
        scenario = s2,
        plot_type = "rbf",
        plot_title = f"Relative bottleneck frequency for {s2[4:6]} (avg. of 10 runs)",
        layout = layout_s2)

    # S3 (two major bottlenecks)
    layout_s3 = ["□–□–■–□–■–□–□", 
                "□–■–□–□–□–■–□", 
                "■–□–□–□–□–□–■"]
    df_s3 = pd.DataFrame()
    for i in range(1, 4):
        s3 = f"12k_S3-{i}"
        df = calc_bottleneck_frequency(s3)
        df_s3[s3] = df["mean"]
    plot_multiple_scenarios(
        df = df_s3, 
        scenario = s3,
        plot_type = "rbf",
        plot_title = f"Relative bottleneck frequency for {s3[4:6]} (avg. of 10 runs)",
        layout = layout_s3)


    ### Relative botttleneck severity (comparison) ###

    # S1 (change in variabilities)
    layout_s1 = ["□–□–□–□–□–□–□ (low var.)",
                "□–□–□–□–□–□–□ (med. var)",
                "□–□–□–□–□–□–□ (high var.)"]
    df_s1 = pd.DataFrame()
    for i in range(1, 4):
        s1 = f"12k_S1-{i}"
        df = calc_bottleneck_severity(s1)
        df_s1[s1] = df["mean"]
    plot_multiple_scenarios(
        df = df_s1, 
        scenario = s1,
        plot_type = "rbs",
        plot_title = f"Relative bottleneck severity for {s1[4:6]} (avg. of 10 runs)",
        layout = layout_s1)

    # S2 (one major bottleneck)
    layout_s2 = ["□–■–□–□–□–□–□", 
                "□–□–□–■–□–□–□", 
                "□–□–□–□–□–■–□"]
    df_s2 = pd.DataFrame()
    for i in range(1, 4):
        s2 = f"12k_S2-{i}"
        df = calc_bottleneck_severity(s2)
        df_s2[s2] = df["mean"]
    plot_multiple_scenarios(
        df = df_s2, 
        scenario = s2,
        plot_type = "rbs",
        plot_title = f"Relative bottleneck severity for {s2[4:6]} (avg. of 10 runs)",
        layout = layout_s2)

    # S3 (two major bottlenecks)
    layout_s3 = ["□–□–■–□–■–□–□", 
                "□–■–□–□–□–■–□", 
                "■–□–□–□–□–□–■"]
    df_s3 = pd.DataFrame()
    for i in range(1, 4):
        s3 = f"12k_S3-{i}"
        df = calc_bottleneck_severity(s3)
        df_s3[s3] = df["mean"]
    plot_multiple_scenarios(
        df = df_s3, 
        scenario = s3,
        plot_type = "rbs",
        plot_title = f"Relative bottleneck severity for {s3[4:6]} (avg. of 10 runs)",
        layout = layout_s3)
