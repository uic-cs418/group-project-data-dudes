import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Viz 1
def viz1(
    df: pd.DataFrame,
    selected_inds: list[float] = None,
    selected_names: list[str] = None
) -> pd.DataFrame:
    if selected_inds is None:
        selected_inds = [2.0, 9.0, 3.0, 8.0, 6.0, 7.0]
    if selected_names is None:
        selected_names = ["Healthcare","Information/Tech","Manufacturing","Govt/Military","Finance","Transportation"]

    attitude_counts = {name: {} for name in selected_names}

    for code, name in zip(selected_inds, selected_names):
        subset = df[df['INDUSTRYCOMBO_W119'] == code]
        total = len(subset)
        if total > 0:
            attitude_counts[name]["Excited"]   = 100 * (subset['CNCEXC_W119'] == 1.0).mean()
            attitude_counts[name]["Concerned"] = 100 * (subset['CNCEXC_W119'] == 2.0).mean()
            attitude_counts[name]["Equal"]     = 100 * (subset['CNCEXC_W119'] == 3.0).mean()

    attitude_df = pd.DataFrame(attitude_counts).T
    return attitude_df

def plot_heatmap(
    attitude_df: pd.DataFrame,
    figsize: tuple[int, int] = (8, 4),
    cmap: str = "Blues"
) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        attitude_df,
        annot=attitude_df.round(0).astype(int),
        fmt="d",
        cmap=cmap,
        cbar_kws={'label': 'Percentage of respondents'}
    )
    plt.title("AI Attitudes (Excited vs. Concerned) by Industry")
    plt.xlabel("Feeling about AI in Daily Life")
    plt.ylabel("Industry")
    plt.tight_layout()
    plt.show()

# Viz 2
def viz2(working, industry_names, name_map):
    use_freq = {}
    for code, group in working.groupby('INDUSTRYCOMBO_W119'):
        pct_daily = 100 * group['USEAI_W119'].isin([1.0, 2.0, 3.0]).mean()
        use_freq[float(code)] = pct_daily

    plot_data = []
    for code, pct in use_freq.items():
        name = industry_names.get(code)
        if name in name_map.index:
            plot_data.append((name_map.loc[name], pct, name))

    x_vals = [pt[0] for pt in plot_data]
    y_vals = [pt[1] for pt in plot_data]
    labels = [pt[2] for pt in plot_data]
    return x_vals, y_vals, labels

def plot_scatter(x_vals, y_vals, labels,
                          figsize=(7, 4),
                          color='purple',
                          point_size=80,
                          alpha=0.7):
    plt.figure(figsize=figsize)
    plt.scatter(x_vals, y_vals, color=color, s=point_size, alpha=alpha)
    plt.title("AI Knowledge vs. AI Usage by Industry")
    plt.xlabel("Average AI Knowledge Score (0–6)")
    plt.ylabel("% Using AI Daily or More")

    for x, y, label in zip(x_vals, y_vals, labels):
        short = label.split('/')[0]
        plt.text(x + 0.03, y + 0.5, short, fontsize=9)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Viz 3
def viz3(working, industry_names, figsize=(7, 4)):
    name_map = (
        working
        .groupby('INDUSTRYCOMBO_W119')['AIWRKH4_W119']
        .apply(lambda x: (x == 1.0).mean() * 100)
    )
    name_map.index = name_map.index.map(industry_names)
    name_map = name_map.sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    positions = list(range(len(name_map)))
    ax.hlines(y=positions, xmin=0, xmax=name_map.values,
              color='lightgray', linewidth=2)
    ax.scatter(name_map.values, positions,
               color='steelblue', s=100, zorder=3)
    for pos, pct in zip(positions, name_map.values):
        ax.text(pct + 1, pos, f"{pct:.0f}%", va='center')
    ax.set_yticks(positions)
    ax.set_yticklabels(name_map.index)
    ax.set_xlabel("Percentage Who Would Apply (%)")
    ax.set_ylabel("Industry")
    ax.set_title("Willingness to Apply to AI-Hiring Employers by Industry")
    ax.set_xlim(0, 100)
    plt.tight_layout()

    return fig, ax
