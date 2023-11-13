import os
import re

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import scipy

from dataset import week_2_season
from model import triwei, get_triwei_params

plt.rcParams.update({'font.size': 13})  # Make text from plots bigger
plt.rcParams.update({'font.weight': 'bold'})


def plot_triwei(x, result_path):
    """
    Plot original data points together with fitter triwei points
    :param x: Dataframe with original and fitted data
    :param result_path: Folder where to store the plot
    """
    raw, fit = x.iloc[:, 0], x.iloc[:, 1]       # 1st column original (raw), 2nd column fit
    k, l, mu1, P1, mu2, P2, mu3, P3, B = get_triwei_params(raw)      # Fit data and get best parameters k->k, l->lambda..
    xi = np.linspace(0, len(x) - 1, len(x))         # Convert x-axis time to units 0,1,2,3...
    x['wei1'] = triwei(xi, k, l, mu1, P1, mu2, 0, mu3, 0, B)    # To get 1st weibull, "deactivate" 2nd and 3rd weibull multiplying them by 0
    x['wei2'] = triwei(xi, k, l, mu1, 0, mu2, P2, mu3, 0, B)    # To get 2nd weibull, "deactivate" 1st and 3rd weibull multiplying them by 0
    x['wei3'] = triwei(xi, k, l, mu1, 0, mu2, 0, mu3, P3, B)    # To get 3rd weibull, "deactivate" 1st and 2nd weibull multiplying them by 0

    title = f"G: {round(k / l, 2)} K: {round(k, 2)} L: {round(l, 2)} M1: {round(mu1, 2)} P1: {round(P1, 2)} | M2: {round(mu2, 2)} P2: {round(P2, 2)} | M3: {round(mu3, 2)} P3: {round(P3, 2)} | B: {round(B, 2)}"
    x.plot(figsize=(20, 10), lw=3, title=title)     # plot the raw and fit datapoints

    wei1 = x['wei1'].values.squeeze()
    peaks, _ = scipy.signal.find_peaks(wei1)
    plt.plot(peaks, wei1[peaks], "x", color='black', lw=20, ms=20, label='wei1 peak')   # plot 1st weibull peak

    wei3 = x['wei3'].values.squeeze()
    peaks, _ = scipy.signal.find_peaks(wei3)
    plt.plot(peaks, wei3[peaks], "x", color='black', lw=20, ms=20, label='wei3 peak')   # plot 3rd weibull peak

    instance = str(x.index[0][0:3])
    instance = re.sub('([0-9])*:([0-9])*', '', instance)    # Remove bad/special charachters from title
    instance = re.sub('[^a-zA-Z0-9]', '_', instance)
    instance = re.sub('_+', '_', instance)
    fname = x.name + instance + ".png"
    plt.savefig(os.path.join(result_path, fname))
    plt.close()


def plot_radial(df, col, folder, xname, to_clock=False):
    """
    Plot polar annual plot
    :param df:
    :param col:
    :param folder:
    :param xname:
    :param to_clock:
    :return:
    """
    if not to_clock:
        fig = px.bar_polar(df, r=col, theta='week', color='season',
                           color_discrete_map={"Winter": "#33bbee", "Spring": "#009988", "Summer": "#ee3377",
                                               "Fall": "#ee7733"},
                           range_r=[df[col].min() - 0.01, df[col].max() + 0.01])
    else:
        print("\n" + col + " summary")
        df = df.pivot_table(index=['week', 'activity'], columns='season', values=col).groupby('activity').mean()
        df.loc[slice("EveningGrad", "MorningPeak")] = df.loc[slice("EveningGrad", "MorningPeak")].apply(lambda x: x.apply(lambda g: idx_to_clock(g, 3)))
        df.loc[slice("ActivityGrad", "ActivityPeak")] = df.loc[slice("ActivityGrad", "ActivityPeak")].apply(lambda x: x.apply(lambda g: idx_to_clock(g, 0)))
        df = df.reindex(sorted(df.columns), axis=1)
        print(df)
        return

    fig.update_layout(width=1000, height=1000)
    fig.update_layout(
        title=str(col),
        legend_title="Season",
        font=dict(family="Courier New, monospace", size=20, color="RebeccaPurple")
    )

    epath = os.path.join(folder, xname)
    os.makedirs(epath, exist_ok=True)

    fig.write_image(os.path.join(epath, f"{col}.png"), engine='kaleido')
    df.to_csv(os.path.join(epath, f"{col}.csv"))


def idx_to_clock(x, offset):
    """
    Convert from Evening/Morning+Peak/Grad x axis units to HH:MM time
    :param x: x axis units
    :param offset: 3 (due to 3AM filtering) for timestamps (f.eks. EveningPeak), 0 for time lenght (f.eks. ActivityGrad)
    :return: HH:MM time
    """
    x = round(x)
    h = x // 60  # Integer division 60 min=1h, HH
    m = x % 60  # Mod, remainder of division, in minutes MM
    h = str(int(h + offset)).zfill(2)  # fill zeroes in front like 3h -> 03h
    m = str(m).zfill(2)  # fill zeroes in fron like 5min -> 05min
    s = f"{h}:{m}"  # f.eks 03:05
    return s


def plot_year_ave(df, exp_path, xname, to_clock=False):
    """
    Plot yearly average activity
    :param df:
    :param exp_path:
    :param xname:
    :param to_clock:
    :return:
    """
    df = df.loc[1:52, :]        # Discard week 53 due to difference in year lengths
    df = df.reset_index()
    df = df.reindex(index=np.roll(df.index, 7))     # Shift rows so that they are properly displayed in the plot
    col = df.columns[-1]

    if col in ['North', "Norge", "Finland", "Sverige"]:
        df['season'] = week_2_season(df['week'], "North")          # Assign season to weeks according to hemisphere
    else:
        df['season'] = week_2_season(df['week'], "South")          # Assign season to weeks according to hemisphere

    df['week'] = df['week'].astype('string')    # Assign string type to week numbers so that they are properly displayed in the plot
    plot_radial(df, col, exp_path, xname, to_clock=to_clock)


def plot_seasons(df, exp_path):
    """
    Plot yearly activity
    :param df:
    :param exp_path:
    :return:
    """
    df = df.sort_index(level=['year', 'week'])
    df.plot(figsize=(30, 20))
    df = df.reset_index()
    title = df['activity'].head(1)[0]
    plt.title(title)

    epath = os.path.join(exp_path, df['activity'][0])
    os.makedirs(epath, exist_ok=True)
    plt.savefig(os.path.join(epath, title+"_seasonal_wave.png"))
    plt.close()
