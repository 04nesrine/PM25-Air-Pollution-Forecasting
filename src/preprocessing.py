import pandas as pd
import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree

def load_datasets(data_dir):
    """
    Load the original PM2.5 and GEE datasets.
    """

    data_dir = Path(data_dir)

    pm25 = pd.read_csv(data_dir / "pm25.csv")
    gee = pd.read_csv(data_dir / "gee_features_daily.csv")
    combined = pd.read_csv(data_dir / "combined_gee_pm25.csv")

    return pm25, gee, combined

def inspect_datasets(pm25, gee, combined):
    """
    Display basic information about the datasets.
    """

    datasets = {
        "PM2.5": pm25,
        "GEE": gee,
        "Combined": combined
    }

    for name, df in datasets.items():
        print(f"\n===== {name} Dataset =====")

        print("Shape:", df.shape)
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())

def clean_pm25_data(combined):
    """
    Remove invalid PM2.5 observations.
    Valid values must be greater than 0 and lower than 500.
    """

    original_count = len(combined)

    cleaned = combined[
        (combined["pm25"] > 0) &
        (combined["pm25"] < 500)
    ].copy()

    removed_count = original_count - len(cleaned)

    print("===== PM2.5 Cleaning Report =====")
    print("Original observations:", original_count)
    print("Cleaned observations:", len(cleaned))
    print("Removed observations:", removed_count)

    return cleaned

def analyze_missing_values(df):
    """
    Calculate missing values count and percentage for each column.
    """

    missing_report = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    return missing_report

def detect_outliers_iqr(df, columns):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    Does not remove any observations.
    """

    results = []

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        )

        results.append({
            "variable": col,
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": round(
                outliers.mean() * 100, 2
            ),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

    return pd.DataFrame(results)

def descriptive_statistics(df, columns):
    """
    Calculate descriptive statistics for selected numerical columns.
    """

    statistics = df[columns].describe().T

    statistics["skewness"] = df[columns].skew()
    statistics["kurtosis"] = df[columns].kurtosis()

    return statistics.round(4)

def correlation_analysis(df, columns):
    """
    Calculate Pearson and Spearman correlation matrices.
    """

    pearson_corr = df[columns].corr(method="pearson")
    spearman_corr = df[columns].corr(method="spearman")

    return pearson_corr, spearman_corr

def map_sites_to_gee(site_summary, gee, threshold=0.25):
    """
    Map each PM2.5 physical site to its nearest GEE ID.
    Sites outside the distance threshold are flagged.
    """

    gee_id_coords = (
        gee.groupby("id")
        .agg(
            gee_lat=("lat", "mean"),
            gee_lon=("lon", "mean")
        )
        .reset_index()
    )

    gee_tree = cKDTree(
        gee_id_coords[["gee_lat", "gee_lon"]].to_numpy()
    )

    distances, indices = gee_tree.query(
        site_summary[["latitude", "longitude"]].to_numpy(),
        k=1
    )

    site_mapping = site_summary.copy()

    site_mapping["gee_id"] = (
        gee_id_coords.iloc[indices]["id"].to_numpy()
    )

    site_mapping["distance_deg"] = distances
    site_mapping["within_threshold"] = distances <= threshold

    return site_mapping

def aggregate_daily_pm25(pm25_sites_valid):
    """
    Aggregate PM2.5 observations by physical site and date.
    Multiple measurements on the same day are averaged.
    """

    pm25_sites_valid = pm25_sites_valid.copy()

    pm25_sites_valid["date"] = pd.to_datetime(
        pm25_sites_valid["date"]
    ).dt.normalize()

    pm25_daily = (
        pm25_sites_valid
        .groupby(["site_id", "date"], as_index=False)
        .agg(
            sensor_name=("sensor_name", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            pm25=("pm25", "mean"),
            gee_id=("gee_id", "first"),
            number_of_measurements=("pm25", "count")
        )
    )

    return pm25_daily

def merge_pm25_gee(pm25_daily, gee):
    """
    Merge daily PM2.5 observations with GEE features
    using date and GEE ID.
    """

    pm25_daily = pm25_daily.copy()
    gee = gee.copy()

    pm25_daily["date"] = pd.to_datetime(
        pm25_daily["date"]
    ).dt.normalize()

    gee["date"] = pd.to_datetime(
        gee["date"]
    ).dt.normalize()

    gee = gee.rename(columns={
        "id": "gee_id",
        "lat": "gee_lat",
        "lon": "gee_lon"
    })

    merged_daily = pm25_daily.merge(
        gee,
        on=["date", "gee_id"],
        how="left",
        validate="many_to_one",
        indicator=True
    )

    return merged_daily

def create_daily_timeline(pm25_daily, gee):
    """
    Create a regular daily timeline for each physical site.
    Preserve GEE features even when PM2.5 is missing.
    """

    pm25_daily = pm25_daily.copy()
    gee = gee.copy()

    pm25_daily["date"] = pd.to_datetime(
        pm25_daily["date"]
    ).dt.normalize()

    gee["date"] = pd.to_datetime(
        gee["date"]
    ).dt.normalize()

    gee = gee.rename(columns={
        "id": "gee_id",
        "lat": "gee_lat",
        "lon": "gee_lon"
    })

    site_dates = (
        pm25_daily.groupby("site_id")
        .agg(
            gee_id=("gee_id", "first"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )
        .reset_index()
    )

    timelines = []

    for _, site in site_dates.iterrows():
        dates = pd.date_range(
            start=site["start_date"],
            end=site["end_date"],
            freq="D"
        )

        timelines.append(pd.DataFrame({
            "site_id": site["site_id"],
            "gee_id": site["gee_id"],
            "date": dates
        }))

    daily_timeline = pd.concat(
        timelines,
        ignore_index=True
    )

    daily_timeline = daily_timeline.merge(
        pm25_daily.drop(columns=["gee_id"]),
        on=["site_id", "date"],
        how="left"
    )

    daily_timeline = daily_timeline.merge(
        gee,
        on=["date", "gee_id"],
        how="left",
        validate="many_to_one"
    )

    daily_timeline = daily_timeline.sort_values(
        ["site_id", "date"]
    ).reset_index(drop=True)

    return daily_timeline

def select_reliable_sites(
    daily_timeline,
    min_pm25_observations=100,
    min_match_rate=40
):
    """
    Select sites based on PM2.5 coverage
    and simultaneous PM2.5-GEE availability.
    """

    data = daily_timeline.copy()

    data["pm25_available"] = data["pm25"].notna()
    data["gee_available"] = data["temperature_celsius"].notna()

    data["both_available"] = (
        data["pm25_available"] & data["gee_available"]
    )

    site_stats = (
        data.groupby("site_id")
        .agg(
            pm25_observations=("pm25_available", "sum"),
            gee_matches=("both_available", "sum")
        )
        .reset_index()
    )

    site_stats["gee_match_rate"] = (
        site_stats["gee_matches"]
        / site_stats["pm25_observations"]
        * 100
    )

    retained_sites = site_stats[
        (site_stats["pm25_observations"] >= min_pm25_observations)
        & (site_stats["gee_match_rate"] >= min_match_rate)
    ].copy()

    return retained_sites

def validate_final_dataset(df):
    """
    Validate the final preprocessed dataset.
    """

    print("===== FINAL DATASET VALIDATION =====")

    print("Shape:", df.shape)

    print("Number of sites:", df["site_id"].nunique())

    print(
        "Duplicate site/date rows:",
        df.duplicated(["site_id", "date"]).sum()
    )

    print("Start date:", df["date"].min())
    print("End date:", df["date"].max())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nPM2.5 invalid values:")

    invalid_pm25 = (
        (df["pm25"] <= 0) |
        (df["pm25"] >= 500)
    )

    print("Invalid observations:", invalid_pm25.sum())



def create_physical_sites(pm25_valid, eps_meters=250):
    """
    Identify physical monitoring sites using DBSCAN
    with the Haversine distance metric.
    """

    from sklearn.cluster import DBSCAN

    pm25_sites = pm25_valid.copy()

    pm25_sites["date"] = pd.to_datetime(
        pm25_sites["date"]
    ).dt.normalize()

    pm25_sites["site_cluster"] = -1

    epsilon = eps_meters / 6371000

    for station, group in pm25_sites.groupby("sensor_name"):

        coords = np.radians(
            group[["latitude", "longitude"]].to_numpy()
        )

        clustering = DBSCAN(
            eps=epsilon,
            min_samples=1,
            metric="haversine"
        )

        labels = clustering.fit_predict(coords)

        pm25_sites.loc[group.index, "site_cluster"] = labels

    pm25_sites["site_id"] = (
        pm25_sites["sensor_name"].astype(str)
        + "_"
        + pm25_sites["site_cluster"].astype(str)
    )

    site_summary = (
        pm25_sites.groupby("site_id")
        .agg(
            sensor_name=("sensor_name", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            observations=("pm25", "count"),
            first_date=("date", "min"),
            last_date=("date", "max")
        )
        .reset_index()
    )

    print("Original stations:", pm25_sites["sensor_name"].nunique())
    print("Physical sites:", site_summary["site_id"].nunique())

    return pm25_sites, site_summary

def save_dataset(df, output_path):
    """
    Save a processed dataset to a CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Dataset saved: {output_path}")