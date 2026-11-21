
from pathlib import Path
import pandas as pd

from src.preprocessing import (
    load_datasets,
    clean_pm25_data,
    create_physical_sites,
    map_sites_to_gee,
    aggregate_daily_pm25,
    merge_pm25_gee,
    create_daily_timeline,
    descriptive_statistics,
    analyze_missing_values,
    detect_outliers_iqr,
    save_dataset
)

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("===== PM2.5 FINAL PREPROCESSING =====")

# 1. Load datasets
pm25, gee, combined = load_datasets(DATA_DIR)

print("\nOriginal PM2.5:", pm25.shape)
print("GEE:", gee.shape)

# 2. Clean invalid PM2.5 observations
pm25_valid = clean_pm25_data(pm25)

# 3. Identify physical monitoring sites
pm25_sites, site_summary = create_physical_sites(pm25_valid)

# 4. Map physical sites to GEE locations
site_mapping = map_sites_to_gee(
    site_summary,
    gee,
    threshold=0.25
)

valid_mapping = site_mapping[
    site_mapping["within_threshold"]
][["site_id", "gee_id"]]

pm25_sites_valid = pm25_sites.merge(
    valid_mapping,
    on="site_id",
    how="inner",
    validate="many_to_one"
)

print("\nValid mapped sites:", valid_mapping["site_id"].nunique())

# 5. Aggregate PM2.5 measurements by site and day
pm25_daily = aggregate_daily_pm25(pm25_sites_valid)

# 6. Merge daily PM2.5 with GEE environmental features
merged_daily = merge_pm25_gee(pm25_daily, gee)

# 7. Create a regular daily timeline
daily_timeline = create_daily_timeline(pm25_daily, gee)

# 8. Prepare exploratory data report
numeric_columns = [
    "pm25",
    "AOD",
    "NO2",
    "CO",
    "O3",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v"
]

statistics = descriptive_statistics(
    merged_daily,
    numeric_columns
)

missing = analyze_missing_values(merged_daily)

outliers = detect_outliers_iqr(
    merged_daily,
    numeric_columns
).set_index("variable")

exploratory_data = statistics.join(missing).join(outliers)


# 9. Prepare temporal exploration report
environmental_columns = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

timeline = daily_timeline.copy()
timeline["date"] = pd.to_datetime(timeline["date"])
timeline["year"] = timeline["date"].dt.year

# Check whether all environmental variables are available
timeline["complete_environmental_day"] = (
    timeline[environmental_columns].notna().all(axis=1)
)

exploratory_temporal = (
    timeline.groupby("year")
    .agg(
        calendar_days=("date", "count"),
        sites=("site_id", "nunique"),
        pm25_observations=("pm25", "count"),
        missing_pm25_days=("pm25", lambda x: x.isna().sum()),
        gee_matches=("temperature_celsius", "count"),
        complete_environmental_days=(
            "complete_environmental_day", "sum"
        )
    )
    .reset_index()
)

# 10. Save final outputs
save_dataset(
    merged_daily,
    OUTPUT_DIR / "cleaned_data.csv"
)

save_dataset(
    daily_timeline,
    OUTPUT_DIR / "daily_timeline.csv"
)

save_dataset(
    exploratory_data.reset_index(),
    OUTPUT_DIR / "Exploratory_Data.csv"
)

save_dataset(
    exploratory_temporal,
    OUTPUT_DIR / "Exploratory_temporal.csv"
)

print("\n===== FINAL VALIDATION =====")
print("Cleaned dataset:", merged_daily.shape)
print("Daily timeline:", daily_timeline.shape)
print("Number of sites:", merged_daily["site_id"].nunique())
print("Duplicate site/date rows:",
      merged_daily.duplicated(["site_id", "date"]).sum())

print("\nAll preprocessing outputs were saved in:")
print(OUTPUT_DIR)
