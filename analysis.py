import glob
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import scipy
from bioinfokit.analys import stat
import pingouin as pg
from send2trash import send2trash
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

from dataset import make_dataset, week_2_season
from model import get_triwei_params, triwei, fit_triwei
from plotting import plot_triwei, plot_year_ave, plot_seasons


def activity(x):
    """
    Calculate daily activity
    :param x: Dataframe containing one day data
    :return: Activity from peak (minutes), Activity from gradients (minutes), x_value 1st peak, x_value 2nd peak, x_value min_gradient of 3rd weibull, x_value max gradient of 1st weibull
    """
    k, l, mu1, P1, mu2, P2, mu3, P3, B = get_triwei_params(x)  # Fit data and get best parameters k->k, l->lambda..
    xi = np.linspace(0, len(x) - 1, len(x))  # Convert x-axis time to units 0,1,2,3...
    wei1 = triwei(xi, k, l, mu1, P1, mu2, 0, mu3, 0, B)  # To get 1st weibull, "deactivate" 2nd and 3rd weibull multiplying them by 0
    wei3 = triwei(xi, k, l, mu1, 0, mu2, 0, mu3, P3, B)  # To get 3rd weibull, "deactivate" 1st and 2nd weibull multiplying them by 0
    peak1, _ = scipy.signal.find_peaks(wei1)  # Find the peak of the 1st weibull
    peak3, _ = scipy.signal.find_peaks(wei3)  # Find the peak of the 3rd weibull
    x_evening_peak = peak3[0] * 4   # Multiply per 4 minutes to get real time, because it was resampled before from 8 to 4 minutes
    x_morning_peak = peak1[0] * 4
    x_evening_grad = np.argmin(np.gradient(wei3)) * 4  # Steepest negative slope of 3rd weibull
    x_morning_grad = np.argmax(np.gradient(wei1)) * 4  # Steepest positive slope of 1st weibull
    return (x_evening_peak - x_morning_peak), (x_evening_grad - x_morning_grad), x_evening_peak, x_morning_peak, x_evening_grad, x_morning_grad


def welch_games(df, title):
    title = str.upper(df.name + "_" + title)
    # perform Welch's ANOVA
    print("\nAnova WELCH " + title)
    print(pg.welch_anova(dv='value', between='season', data=df))

    print("\nPostHoc Games-Howell " + title)
    print(pg.pairwise_gameshowell(data=df, dv='value', between='season').round(4))


def anova_test(df, e_path):
    """
    Prints anova results
    :param df:
    """
    res = stat()  # Initialize object from bioinfokit.analys for later use
    df[['season', 'hemis']] = df[['season', 'hemis']].astype('category')

    title = df.name
    print("\nResults for " + title)

    print("\n" + "Anova NORMAL")
    if len(df['hemis'].unique()) < 2:  # If all countries belong a single hemisphere
        res.anova_stat(df=df[['value', 'season']], res_var='value', anova_model='value~C(season)')
        print(res.anova_summary)
        print("\n")
    else:
        res.anova_stat(df=df[['value', 'season', 'hemis']], res_var='value',
                       anova_model='value~C(season)+C(hemis)+C(season):C(hemis)')
        print(res.anova_summary)
        print("\n")

    try:
        df[df.year < 2023].groupby(["hemis"]).apply(lambda x: welch_games(x, title))  # Remove 2023 for balanced anova
    except:
        df.groupby(["hemis"]).apply(lambda x: welch_games(x, title))  # Remove 2023 for balanced anova

    # res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
    sm.qqplot(res.anova_std_residuals, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    os.makedirs(os.path.join(e_path, title), exist_ok=True)
    plt.savefig(os.path.join(e_path, title, "std_residuals.png"))
    plt.close()

    # histogram
    plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
    plt.xlabel("Residuals")
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(e_path, title, "histogram_residuals.png"))
    plt.close()

    # Shapiro-Wilk test
    w, pvalue = stats.shapiro(res.anova_model_out.resid)
    print("Shapiro w: " + str(w) + " p_value: " + str(pvalue))
    print("\n")

    res.bartlett(df=df, res_var='value', xfac_var='season')
    print("Bartlett")
    print(res.bartlett_summary)
    print("\n ------------------------------------------------------------------ \n")


def tukey_test(df):
    """
    Prints tukey tests
    :param df:
    """
    title = str.upper("_".join(df.name))
    print("Tukey " + title)
    res = stat()  # Initialize object from bioinfokit.analys for later use
    df['season'] = df['season'].astype('category')
    print({c: i for i, c in enumerate(pd.Categorical.from_codes(codes=range(4), dtype=df['season'].dtype))})
    df['season'] = df['season'].cat.codes  # Convert season names to season ID numbers
    res.tukey_hsd(df=df[['value', 'season']], res_var='value', xfac_var='season', anova_model='value~C(season)')
    print(res.tukey_summary)
    print("\n ------------------------------------------------------------------ \n")


def activity_analysis(df, result_path, plot_triwei_fit=False):
    """
    Perform activity calculations for every day (representing week of year)
    :param df: Daily dataframe normalized between 0-1 for each day (week)
    :param result_path: Folder where to store triwei folder plot results
    :param plot_triwei_fit: Whether to plot triwei fitting curves
    :return: Activity dataframe using different methods
    """
    grouper = list({"year", "week"}.intersection(set(df.index.names)))  # Get available levels. Year won't appear in yearly averaged cases.
    if plot_triwei_fit:
        triwei_path = os.path.join(result_path, "triwei_plots")  # Folder to store fitting triwei plot images
        if os.path.exists(triwei_path):
            send2trash(triwei_path)  # Remove plots if they already exist
        os.makedirs(triwei_path)  # Make necessary folders to store plot images

        tri_df = df.groupby(by=grouper).transform(lambda g: fit_triwei(g))  # For every month of year, fit triwei model
        plot_df = pd.concat([df, tri_df], axis=1, keys=['raw', 'fit'])  # Make a dataframe with original and fitted column values for comparison
        plot_df = plot_df.reorder_levels(list(np.roll(np.arange(0, plot_df.columns.nlevels), -1)), axis=1)  # Transform column level order f.eks. raw/North/Finland -> North/Finland/raw
        plot_df.groupby(by=grouper).apply(lambda x: x.groupby(level=-2, axis=1).apply(lambda g: plot_triwei(g, triwei_path)))  # For each week of year and country plot raw and fit

    act_df = df.groupby(by=grouper).apply(lambda x: x.apply(lambda g: activity(g)))  # For each week of year and country get activity metrics
    act_df = act_df.rename_axis(act_df.index.names[:-1] + ["activity"])  # Append level name to index
    act_df.index = act_df.index.set_levels(["ActivityPeak", "ActivityGrad", "EveningPeak", "MorningPeak", "EveningGrad", "MorningGrad"], level="activity")  # Convert numbers to proper labels in activity level
    return act_df


def plot_hist(anova_df, e_path):
    act = anova_df['activity'].head(1).values[0]
    anova_df['value'].hist(bins=30)
    fname = str.upper("_".join(anova_df.head(1)[['hemis', 'season']].values[0]))
    os.makedirs(os.path.join(e_path, act), exist_ok=True)
    plt.title(fname)
    plt.savefig(os.path.join(e_path, act, fname + "_histogram.png"))
    plt.close()
    anova_df.to_csv(os.path.join(e_path, act, fname + "_histogram.csv"))


def run_anova_tukey(act_df, e_path):
    #anova_df_old = act_df.groupby(['week', 'activity']).mean().dropna().reset_index().melt(id_vars=['week', 'activity'])  # Reformat dataframe for anova table input
    if "year" in act_df.index.names:
        anova_df = act_df.reset_index().melt(id_vars=['year', 'week', 'activity'])
    else:
        anova_df = act_df.reset_index().melt(id_vars=['week', 'activity'])

    anova_df['season'] = anova_df['hemis'].mask(anova_df['hemis'] == "North", week_2_season(anova_df['week'],"North"))  # Add season column according to hemispheres
    anova_df['season'] = anova_df['season'].mask(anova_df['season'] == "South", week_2_season(anova_df['week'],"South"))  # South hemisphere filled with opposite seasons than North, for each week
    anova_df.groupby(["activity"]).apply(lambda x: anova_test(x, e_path))  # Run anova per activity type f.eks. Weekly, ActivityGrad, ActivityPeak
    anova_df.groupby(["hemis", "activity"]).apply(lambda x: tukey_test(x))  # Run tukey per hemisphere and activity type to find out effects of season
    anova_df.groupby(["activity", "hemis", "season"]).apply(lambda x: plot_hist(x, e_path))


def run_radial_plot(act_df, exp_path, geo=False, to_clock=False):
    """
    Plots annual radial figures
    :param act_df: Dataframe with activity metrics
    :param exp_path: Folder where to store plots
    :param geo: Whether to make plots also by country
    :param to_clock: Whether to transform axis to HH:MM
    """
    plot_df = act_df.groupby(['week', 'activity']).mean().dropna()  # Make mean across years and by activity type
    if to_clock:
        grad_or_peak = plot_df.index.get_level_values('activity').isin(["null"])
        plot_df = plot_df.groupby(grad_or_peak)
    else:
        plot_df = plot_df.groupby(axis=0, level="activity")

    plot_df.apply(lambda x: x.groupby("hemis", axis=1).mean().apply(lambda g: plot_year_ave(g, exp_path, x.name if not isinstance(x.name, bool) else g.name, to_clock)))  # Plot by hemisphere
    if geo:
        plot_df.apply(lambda x: x.groupby("geo", axis=1).mean().apply(lambda g: plot_year_ave(g, exp_path, x.name if not isinstance(x.name, bool) else g.name, to_clock)))  # Plot by country


def run_weekly_experiment(df, exp_path):
    """
    Run anova, tukey analysis and plot results to export path for weekly data.
    :param df: Dataframe with weekly data
    :param exp_path: Export path (folder)
    """
    if os.path.exists(exp_path):
        send2trash(exp_path)
    os.makedirs(exp_path)

    with open(os.path.join(exp_path, 'results.txt'), 'w') as f:
        with redirect_stdout(f):  # Save to file results.txt whatever is printed inside this "with"
            print(exp_path)

            plot_df = df.groupby(axis=1, level="hemis").mean()
            plot_df.groupby(axis=0, level="activity").apply(lambda x: plot_seasons(x, exp_path))

            run_anova_tukey(df, exp_path)
            run_radial_plot(df, exp_path, True)


def run_hourly_experiment(df, exp_path):
    """
    Run anova, tukey analysis and plot results to export path for hourly (minutely) data.
    :param df: Dataframe dataset
    :param exp_path: Export path for saving results (folder)
    """
    # idx_grouper is used to group indexes for computing means for each group
    # col_grouper is used to group columns for computing means for each group
    for idx_grouper, col_grouper, experiment in [(None, ['hemis', 'geo'], "Activity"),
                                                 (None, ['hemis'], "Activity from hemisphere averaged"),

                                                 (['week', 'time'], ['hemis', 'geo'], "Activity from yearly averaged"),
                                                 (['week', 'time'], ['hemis'],
                                                  "Activity from yearly and hemisphere averaged")]:

        e_path = os.path.join(exp_path, experiment)
        if os.path.exists(e_path):
            send2trash(e_path)
        os.makedirs(e_path)

        with open(os.path.join(e_path, 'results.txt'), 'w') as f:
            with redirect_stdout(f):  # Save to file results.txt whatever is printed inside this "with"
                print(e_path)
                mean_df = df.groupby(axis=1, level=col_grouper).mean()  # Group by columns to make means f.eks by hemisphere
                if idx_grouper:
                    mean_df = mean_df.groupby(idx_grouper).mean().dropna()  # Group by indexes to make means f.eks by week across years
                    mean_df.index = mean_df.index.remove_unused_levels()
                norm_df = mean_df.transform(lambda x: x / x.max())  # Normalize data between 0 and 1 for triwei fitting
                act_df = activity_analysis(norm_df, e_path, plot_triwei_fit=False)

                if not idx_grouper:  # If no index grouper, meaning we have data of several years in act_df, then plot history scattered
                    plot_df = act_df.groupby(axis=1, level="hemis").mean()
                    plot_df.groupby(axis=0, level="activity").apply(lambda x: plot_seasons(x, e_path))

                run_anova_tukey(act_df, e_path)
                run_radial_plot(act_df, e_path, "geo" in col_grouper)  # "geo" in col_grouper translates to whether data per country yet exists or was already averaged
                run_radial_plot(act_df, e_path, "geo" in col_grouper, to_clock=True)  # to_clock=True for timestamps we want HH:MM in x axis of the plot

# If the program freezes, check package kaleido
if __name__ == "__main__":
    porn_csv_files = glob.glob(os.path.join("GData", "porn_keywords", "**/*.csv"), recursive=True)  # Get all excel filenames in any folder under GData/porn_keywords
    porn_dataset_path = os.path.join("datasets", "porn_keywords")  # Dataset folder
    df = make_dataset(porn_csv_files, porn_dataset_path, update=True)  # Make dataset from Google Trend query results
    df['activity'] = 'Weekly'  # Activity type weekly instead of ActivityPeak, ActivityGrad. No need for triwei in this case
    df.set_index([df.index, 'activity'], inplace=True)  # Insert activity level so that there are 3 levels and it becomes compatible with plotting functions using 3 levels
    run_weekly_experiment(df, os.path.join("results", "porn_keywords"))

    for cat in ["news_category", "artsNentertainment_category"]:
        cat_csv_files = glob.glob(os.path.join("GData", cat, "**/*.csv"), recursive=True)
        cat_dataset_path = os.path.join("datasets", cat)
        df = make_dataset(cat_csv_files, cat_dataset_path, update=True)
        run_hourly_experiment(df, os.path.join("results", cat))
