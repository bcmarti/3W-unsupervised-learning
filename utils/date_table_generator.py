import os
import re
import pandas as pd
from collections import defaultdict

DATASET_PATH = "dataset"
OUTPUT_PATH = "well_date_ranges.csv"

# well_number -> {start, end, obs_count}
well_data = defaultdict(lambda: {"start": None, "end": None, "obs_count": 0})

def update_well(well_num, date_min, date_max, row_count):
    entry = well_data[well_num]
    if entry["start"] is None or date_min < entry["start"]:
        entry["start"] = date_min
    if entry["end"] is None or date_max > entry["end"]:
        entry["end"] = date_max
    entry["obs_count"] += row_count

files_processed = 0
files_failed = 0

all_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for filename in sorted(files):
        if filename.startswith("WELL") and filename.endswith(".parquet"):
            all_files.append((root, filename))

print(f"Found {len(all_files)} WELL parquet files. Processing...")

for i, (root, filename) in enumerate(all_files, 1):
    match = re.match(r"WELL-(\d+)", filename)
    if not match:
        continue

    well_num = int(match.group(1))
    filepath = os.path.join(root, filename)

    try:
        # Read only the index (timestamp) — skips all sensor columns
        df = pd.read_parquet(filepath, columns=[])
        index = df.index

        if not isinstance(index, pd.DatetimeIndex):
            index = pd.to_datetime(index, errors="coerce")

        index = index.dropna()
        if len(index) == 0:
            raise ValueError("Empty or unparseable timestamp index")

        update_well(well_num, index.min(), index.max(), len(index))
        files_processed += 1

        if i % 50 == 0 or i == len(all_files):
            print(f"  [{i}/{len(all_files)}] processed...")

    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        files_failed += 1

# Totals
total_obs = sum(v["obs_count"] for v in well_data.values())

# Build output CSV
rows = []
for well_num in sorted(well_data):
    entry = well_data[well_num]
    obs = entry["obs_count"]
    rows.append({
        "well":            f"WELL-{well_num:05d}",
        "start_date":      entry["start"].strftime("%Y-%m-%d %H:%M:%S") if entry["start"] else "",
        "end_date":        entry["end"].strftime("%Y-%m-%d %H:%M:%S") if entry["end"] else "",
        "obs_count":       obs,
        "obs_share_pct":   round(obs / total_obs * 100, 4) if total_obs > 0 else 0.0,
    })

output_df = pd.DataFrame(rows, columns=["well", "start_date", "end_date", "obs_count", "obs_share_pct"])
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDone! {files_processed} files processed, {files_failed} failed.")
print(f"Total observations across all wells: {total_obs:,}")
print(f"Output saved to: {OUTPUT_PATH}\n")
print(output_df.to_string(index=False))
