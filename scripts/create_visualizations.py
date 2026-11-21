
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "preprocessing"
FIGURES_DIR = DATA_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load processed data
df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded:", df.shape)

# Variables used in the analysis
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

numeric_columns = ["pm25"] + environmental_columns


# ==========================================
# GRAPH 1: PM2.5 DISTRIBUTION
# ==========================================

plt.figure(figsize=(9, 5))

plt.hist(
    df["pm25"].dropna(),
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of Daily PM2.5 Concentrations")
plt.xlabel("PM2.5 concentration")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_pm25_distribution.png", dpi=300)
plt.close()


# ==========================================
# GRAPH 2: PM2.5 BOXPLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.boxplot(
    df["pm25"].dropna(),
    vert=True
)

plt.title("Boxplot of PM2.5 Concentrations")
plt.ylabel("PM2.5 concentration")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_pm25_boxplot.png", dpi=300)
plt.close()


# ==========================================
# GRAPH 3: CORRELATION MATRIX
# ==========================================

correlation = df[numeric_columns].corr(method="pearson")

plt.figure(figsize=(11, 8))

image = plt.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(image, label="Pearson correlation")

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

for i in range(len(numeric_columns)):
    for j in range(len(numeric_columns)):
        value = correlation.iloc[i, j]
        if pd.notna(value):
            plt.text(
                j, i, f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

plt.title("Correlation Matrix of PM2.5 and Environmental Variables")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_correlation_matrix.png", dpi=300)
plt.close()


# ==========================================
# GRAPH 4: MISSING VALUES
# ==========================================

missing_percentage = (
    df[numeric_columns].isnull().mean() * 100
).sort_values(ascending=False)

plt.figure(figsize=(10, 5))

plt.bar(
    missing_percentage.index,
    missing_percentage.values,
    edgecolor="black"
)

plt.title("Percentage of Missing Values by Variable")
plt.xlabel("Variable")
plt.ylabel("Missing values (%)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_missing_values.png", dpi=300)
plt.close()


# ==========================================
# GRAPH 5: MONTHLY PM2.5 EVOLUTION
# ==========================================

monthly_pm25 = (
    df.set_index("date")["pm25"]
    .resample("MS")
    .mean()
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_pm25.index,
    monthly_pm25.values,
    marker=".",
    linewidth=1.5
)

plt.title("Monthly Evolution of Average PM2.5 Concentration")
plt.xlabel("Date")
plt.ylabel("Average PM2.5 concentration")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_monthly_pm25_evolution.png", dpi=300)
plt.close()


print("\n===== VISUALIZATIONS COMPLETED =====")
print("Figures saved in:", FIGURES_DIR)

for file in sorted(FIGURES_DIR.glob("*.png")):
    print(file.name)
