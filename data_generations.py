import pandas as pd
import geopandas as gpd

# 1. Load the crash data (shapefile with geometries)
crashes = gpd.read_file("./Crashes_in_DC.shp")

# 2. Load the roadway block data (shapefile with geometries)
roadway = gpd.read_file("./Roadway_Block.shp")

# 3. Basic cleaning
crashes = crashes.dropna(subset=["ROADWAYSEG"])

# Optional: filter by year range
crashes["REPORTDATE"] = pd.to_datetime(crashes["REPORTDATE"])
crashes = crashes[crashes["REPORTDATE"].dt.year.between(2017, 2023)]

# Calculate total severe/fatal injuries per crash
crashes["severe_injuries"] = (
    crashes["MAJORINJUR"].fillna(0) + 
    crashes["MAJORINJ_1"].fillna(0) + 
    crashes["MAJORINJ_2"].fillna(0) +
    crashes["MAJORINJ_3"].fillna(0) +
    crashes["MAJORINJ_4"].fillna(0)
)

crashes["fatal_injuries"] = (
    crashes["FATAL_BICY"].fillna(0) + 
    crashes["FATAL_DRIV"].fillna(0) + 
    crashes["FATAL_PEDE"].fillna(0) +
    crashes["FATALPASSE"].fillna(0) +
    crashes["FATALOTHER"].fillna(0)
)

# Flag for pedestrian/bike involved
crashes["ped_bike_involved"] = (
    (crashes["TOTAL_BICY"].fillna(0) > 0) | 
    (crashes["TOTAL_PEDE"].fillna(0) > 0)
).astype(int)

# 4. Spatial join: match crashes to nearest roadway segment
# First, ensure both have the same CRS
if crashes.crs != roadway.crs:
    crashes = crashes.to_crs(roadway.crs)

# Perform spatial join to match crashes to roadway segments
crashes_with_road = gpd.sjoin_nearest(crashes, roadway[["ROUTEID", "geometry"]], how="left", max_distance=50)

# Rename the joined ROUTEID column
if "ROUTEID_right" in crashes_with_road.columns:
    crashes_with_road = crashes_with_road.rename(columns={"ROUTEID_right": "ROADWAY_ROUTEID"})
elif "ROUTEID" not in crashes_with_road.columns:
    # If ROUTEID exists but no suffix, it means it wasn't duplicated
    crashes_with_road = crashes_with_road.rename(columns={"ROUTEID": "ROADWAY_ROUTEID"})
else:
    crashes_with_road["ROADWAY_ROUTEID"] = crashes_with_road["ROUTEID"]

# 5. Aggregate crashes to roadway block level
agg = crashes_with_road.groupby("ROADWAY_ROUTEID").agg(
    crash_count=("OBJECTID", "count"),
    severe_count=("severe_injuries", "sum"),
    fatal_count=("fatal_injuries", "sum"),
    ped_bike_crashes=("ped_bike_involved", "sum"),
    total_vehicles=("TOTAL_VEHI", "sum"),
    speeding_involved=("SPEEDING_I", "sum")
).reset_index()

# 6. Create derived crash features
years = crashes["REPORTDATE"].dt.year.nunique()
agg["crash_count_per_year"] = agg["crash_count"] / years
agg["severe_share"] = (agg["severe_count"] + agg["fatal_count"]) / agg["crash_count"]
agg["ped_bike_share"] = agg["ped_bike_crashes"] / agg["crash_count"]
agg["speeding_share"] = agg["speeding_involved"].fillna(0) / agg["crash_count"]

# 7. Prepare roadway features
# Aggregate roadway features by ROUTEID
roadway_features = roadway.groupby("ROUTEID").agg({
    "TOTALTRAVE": "max",      # Total travel lanes
    "SPEEDLIMIT": "first",     # Speed limit
    "DCFUNCTION": "first",     # Functional class
    "TOTALBIKEL": "max",       # Total bike lanes
    "AADT": "max"              # Annual average daily traffic
}).reset_index()

# Clean speed limits (convert to numeric)
roadway_features["speed_limit"] = pd.to_numeric(roadway_features["SPEEDLIMIT"], errors="coerce")

# Fill missing values and rename columns
roadway_features = roadway_features.rename(columns={
    "TOTALTRAVE": "n_lanes",
    "DCFUNCTION": "func_class",
    "TOTALBIKEL": "bike_lanes",
    "AADT": "traffic_volume"
})

roadway_features["n_lanes"] = roadway_features["n_lanes"].fillna(2)
roadway_features["speed_limit"] = roadway_features["speed_limit"].fillna(25)
roadway_features["bike_lanes"] = roadway_features["bike_lanes"].fillna(0)
roadway_features["traffic_volume"] = roadway_features["traffic_volume"].fillna(0)

# 8. Merge crash data with roadway features
data = agg.merge(roadway_features[["ROUTEID", "n_lanes", "speed_limit", "func_class", "bike_lanes", "traffic_volume"]], 
                 left_on="ROADWAY_ROUTEID", right_on="ROUTEID", how="left")

# 9. One-hot encode functional class
if "func_class" in data.columns:
    func_dummies = pd.get_dummies(data["func_class"], prefix="func", dummy_na=True)
    data = pd.concat([data, func_dummies], axis=1)
