import os

import pandas as pd
from send2trash import send2trash


def week_2_season(weeks, hemis):
    """
    Converts week numbers of the year 0-53 into season names
    :param weeks:
    :param hemis:
    :return:
    """
    if hemis == "North":
        return pd.cut((weeks + 7) % 53, 4, labels=["Winter", "Spring", "Summer", "Fall"])  # Seasons centered around solstice and equinox
    else:
        return pd.cut((weeks + 7) % 53, 4, labels=["Summer", "Fall", "Winter", "Spring"])  # If hemisphere==South then opposite seasons


def filter_wday(df, wday):
    """
    Filter results from Google Trends query by day of week (wday)
    :param df: Dataframe holding query result (timelined data)
    :param wday: Day of week 0-6 (Monday-Sunday)
    :return: Filtered dataframe according to wday from 3AM to next day 3AM
    """
    idx = df.index
    next_wday = (wday + 1) % 7  # Calculate next day

    day = df[(idx.weekday == wday) & (idx.hour >= 3)]  # Get data from chosen day (wday) from 3AM
    next_day = df[(idx.weekday == next_wday) & (idx.hour <= 3)]  # Get data from next day until 3AM

    next_day.index -= pd.Timedelta(days=1)  # Substract 1 day from next_day index so that we can build a wday of more than 24 hours (3AM to 28PM) so that we can manipulate it in pandas

    day_time = list(map(lambda t: str(t[0]).zfill(2) + ":" + str(t[1]).zfill(2), list(zip(day.index.hour, day.index.minute))))  # Extract from datetime index string hours like HH:MM "03:00", "03:04", "03:12"...
    next_day_time = list(map(lambda t: str(t[0]).zfill(2) + ":" + str(t[1]).zfill(2), list(zip(next_day.index.hour + 24, next_day.index.minute))))  # Extract from datetime index string hours like HH:MM "24:00", "24:04", "24:08"...

    day_year, day_week = day.index.isocalendar()['year'], day.index.isocalendar()['week']
    day.index = pd.MultiIndex.from_arrays([day_year, day_week, day_time], names=["year", "week", "time"])  # Make multilevel index year/week/time

    next_day_year, next_day_week = next_day.index.isocalendar()['year'], next_day.index.isocalendar()['week']
    next_day.index = pd.MultiIndex.from_arrays([next_day_year, next_day_week, next_day_time], names=["year", "week", "time"])  # Make multilevel index year/week/time

    return pd.concat([day, next_day]).sort_index()  # Merge wday from 3AM and next day to 3AM into one dataframe from 03:00 to 27:56


def make_dataset(csv_files, dataset_path, update=False):
    """
    Read csv files downloaded from Google Trends, process all into one pandas dataframe, and save this into pickle file.
    :param csv_files: List of csv filenames containing data from Google Trends.
    :param dataset_path: Folder where to save pickle file.
    :param update: Whether to use current pickle file if it already exists, or create new (discard current if it exists).
    :return: Pandas dataframe
    """
    pickle_path = os.path.join(dataset_path, "dataset.pkl")

    if not update and os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)  # Load existing data from "dataset.pkl" into a pandas dataframe and return it.

    if os.path.exists(dataset_path):
        send2trash(dataset_path)  # If update==True and "dataset.pkl" already exists, delete it, send it to trash.
    os.makedirs(dataset_path)  # Make the folders where "dataset.pkl" is going to be stored into.

    results = []  # Array where many dataframes are going to be stored. Every dataframe comes from one csv file.
    for f in csv_files:  # Iterate every csv file (every result from querying Google Trends)
        df = pd.read_csv(f, header=0)  # Load csv file data into pandas dataframe
        cat = df.columns[0].split(":")[1].strip()  # Extract category name from column title f.eks. Kategori: Alle kategorier  -> "Alle categorier"
        time_delta = df.index[0]  # Extract time frequency from first index f.eks. "Tid" eller "Uke"
        geo = df.iat[0, 0].split(":")[1].strip().replace("(", "").replace(")", "")  # Extract country from first row f.eks. Porn + XNXX: (New Zealand) -> "New Zealand"
        match geo:
            case "Norge":  # In case geo variable matches this country (Norway), do this...
                timezone = "Europe/Oslo"
                hemis = "North"
            case "Finland":
                timezone = "Europe/Helsinki"  # ... else if Finland, this...
                hemis = "North"
            case "Sverige":
                timezone = "Europe/Stockholm"
                hemis = "North"
            case "New Zealand":
                timezone = "Pacific/Auckland"
                hemis = "South"
            case "Victoria":
                timezone = "Australia/Melbourne"
                hemis = "South"
            case _:
                raise NotImplementedError("Timezone not found")

        df = df.drop(df.head(1).index)  # Remove first row f.eks. "Uke": Porn + XNXX
        df.index = pd.to_datetime(df.index)  # Transforms dates in string format to pandas datetime format UTC aware
        if time_delta == "Tid":  # If minutely or hourly data
            new_df = df.set_index(df.index.tz_convert(timezone).tz_localize(None))  # Convert indexes into local time
            if len(new_df.asfreq(freq="8min")) > len(df):  # If time frequency is distorted due to DST starting
                new_df = new_df.asfreq(freq="8min")
                new_df[:len(df)] = df.values  # Discard "bad" values due to DST change and keep using 8min as time resolution
            df = new_df[:len(df)]  # Also make sure "bad" values discared due to DST finishing
        else:  # if weekly
            df.index = df.index.tz_localize(None)  # Make datetime index UTC unaware (allowing simple date operations)

        df.columns = pd.MultiIndex.from_tuples([(hemis, geo, cat)], names=["hemis", "geo", "cat"])  # Make column of multiple levels from hemis to cat, f.eks. "Alle kategorier" inside "New Zealand" inside "South"
        df = df.loc[~df.index.duplicated(keep='first')]  # If there are duplicated indexes due to timezone changes like DST, keep the last one.
        results += [df]  # Append processed query results from Google Trends to the final dataframe to be saved into dataset.pkl.

    df = pd.concat(results).sort_index()  # Concatenate all results into dataframe of multiple columns f.eks "South/New Zealand/Alle kategorier", "North/Norge/Alle kategorier"...
    df = df.groupby(axis=0, level=0).first()  # Share index across columns

    if time_delta == "Tid":
        grouper = df.index.isocalendar()[['year', 'week']].T.values.tolist()
        df = df.astype('float64').groupby(grouper, as_index=False, group_keys=False).resample("4min").interpolate(limit=1, limit_direction="both").dropna()  # Upsample and interpolate index from 8min to 4min to make sure all indexes are shared across countries (columns)
        df = filter_wday(df, 5)  # Filter data by Saturday from 3AM to next day (Sunday) 3AM.
        df = df.groupby(by=["week"], as_index=False, group_keys=False).transform(lambda x: x * (100 / x.max()))  # Make sure each Saturday is in range between 0 and 100 after filtering out data (normalized)

    else:
        df = df.astype('float64').dropna()  # Convert to real number and drop empty rows
        year, week = df.index.isocalendar()['year'], df.index.isocalendar()['week']  # Get years and weeks from datetime index
        df.index = pd.MultiIndex.from_arrays([year, week], names=["year", "week"])  # Convert datetime index to year/week index (multilevel, each year has X weeks)

    df.to_pickle(pickle_path)  # Save dataframe to dataset.pkl file
    return df
