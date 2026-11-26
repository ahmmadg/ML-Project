"""Shared preprocessing module for CairoGo (Pack v03).

Provides leakage-free data repairs and feature engineering for both
model training (Task 3/3b/6) and standalone inference (Task 8 predict.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEGACY_ZONE_MAP = {
    "6th October": "6th October District",
    "Helwan": "Helwan District",
    "Qalyub": "Qalyub District",
}

UNIT_SHIFT_DATE = "2026-08-27"
MILES_TO_KM = 1.60934

LEAKAGE_COLUMNS = [
    "actual_trip_time_min",
    "cancel_reason",
    "driver_arrival_min",
    "trip_rating",
]

REQUEST_TIME_COLUMNS = [
    "trip_id",
    "customer_id",
    "driver_id",
    "request_ts",
    "pickup_zone",
    "dropoff_zone",
    "distance_km",
    "surge_multiplier",
    "est_fare",
    "eta_estimate_min",
    "weather",
    "payment_method",
    "service_level",
]


def clean_requests(
    df: pd.DataFrame,
    fix_zones: bool = True,
    fix_unit_shift: bool = True,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Apply audited data repairs to a raw requests DataFrame."""
    out = df.copy()

    if drop_leakage:
        cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in out.columns]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)

    if fix_zones:
        if "pickup_zone" in out.columns:
            out["pickup_zone"] = out["pickup_zone"].replace(LEGACY_ZONE_MAP)
        if "dropoff_zone" in out.columns:
            out["dropoff_zone"] = out["dropoff_zone"].replace(LEGACY_ZONE_MAP)

    ts = pd.to_datetime(out["request_ts"], errors="coerce")
    if fix_unit_shift and "distance_km" in out.columns:
        shift_mask = ts >= pd.Timestamp(UNIT_SHIFT_DATE)
        out.loc[shift_mask, "distance_km"] = (
            out.loc[shift_mask, "distance_km"] * MILES_TO_KM
        )

    # Informative missingness flag for weather (Task 2)
    out["weather_is_missing"] = out["weather"].isna().astype(int)
    out["weather"] = out["weather"].fillna("MISSING")

    # Request-time calendar features
    out["request_hour"] = ts.dt.hour.fillna(12).astype(int)
    out["request_dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    out["is_weekend"] = out["request_dayofweek"].isin([4, 5]).astype(int)
    out["is_night"] = (
        (out["request_hour"] >= 22) | (out["request_hour"] <= 5)
    ).astype(int)

    # Safe ratio feature
    dist_safe = out["distance_km"].replace(0, np.nan)
    out["fare_per_km"] = (out["est_fare"] / dist_safe).fillna(0.0)
    out["same_zone"] = (
        out["pickup_zone"].astype(str) == out["dropoff_zone"].astype(str)
    ).astype(int)

    return out


def build_features(
    df: pd.DataFrame,
    zones_df: pd.DataFrame | None = None,
    customers_df: pd.DataFrame | None = None,
    drivers_df: pd.DataFrame | None = None,
    fix_zones: bool = True,
    fix_unit_shift: bool = True,
) -> pd.DataFrame:
    """Clean requests and join zone, customer, and driver metadata safely."""
    out = clean_requests(
        df,
        fix_zones=fix_zones,
        fix_unit_shift=fix_unit_shift,
        drop_leakage=True,
    )

    if zones_df is not None:
        pz = zones_df[["zone_name", "zone_type", "has_metro"]].rename(
            columns={
                "zone_name": "pickup_zone",
                "zone_type": "pickup_zone_type",
                "has_metro": "pickup_has_metro",
            }
        )
        dz = zones_df[["zone_name", "zone_type", "has_metro"]].rename(
            columns={
                "zone_name": "dropoff_zone",
                "zone_type": "dropoff_zone_type",
                "has_metro": "dropoff_has_metro",
            }
        )
        out = out.merge(pz, on="pickup_zone", how="left")
        out = out.merge(dz, on="dropoff_zone", how="left")
        out["pickup_zone_type"] = out["pickup_zone_type"].fillna("UNKNOWN")
        out["dropoff_zone_type"] = out["dropoff_zone_type"].fillna("UNKNOWN")
        out["pickup_has_metro"] = out["pickup_has_metro"].fillna(0).astype(int)
        out["dropoff_has_metro"] = out["dropoff_has_metro"].fillna(0).astype(int)

    if customers_df is not None:
        cust = customers_df[
            ["customer_id", "channel", "loyalty_tier", "home_zone"]
        ].rename(columns={"home_zone": "customer_home_zone"})
        out = out.merge(cust, on="customer_id", how="left")
        out["channel"] = out["channel"].fillna("UNKNOWN")
        out["loyalty_tier"] = out["loyalty_tier"].fillna("UNKNOWN")
        out["customer_home_zone"] = out["customer_home_zone"].fillna("UNKNOWN")
        out["is_customer_home_pickup"] = (
            out["pickup_zone"].astype(str) == out["customer_home_zone"].astype(str)
        ).astype(int)

    if drivers_df is not None:
        drv = drivers_df[
            ["driver_id", "vehicle_type", "experience_months", "rating_avg", "home_zone"]
        ].rename(columns={"home_zone": "driver_home_zone"})
        out = out.merge(drv, on="driver_id", how="left")
        out["rating_avg_is_missing"] = out["rating_avg"].isna().astype(int)
        out["vehicle_type"] = out["vehicle_type"].fillna("UNKNOWN")
        out["driver_home_zone"] = out["driver_home_zone"].fillna("UNKNOWN")
        out["is_driver_home_pickup"] = (
            out["pickup_zone"].astype(str) == out["driver_home_zone"].astype(str)
        ).astype(int)

    return out


def time_split(
    df: pd.DataFrame, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation split sorted by request_ts and trip_id."""
    ordered = df.sort_values(["request_ts", "trip_id"]).reset_index(drop=True)
    split_idx = int(len(ordered) * (1.0 - val_frac))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()
