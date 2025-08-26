
import os, time, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import fastparquet

OUT_DIR = "bigdata_dask_demo_out"
PARQUET_DIR = os.path.join(OUT_DIR, "parquet_dataset")
N_ROWS = 2_000_000       
N_PARTITIONS = 40        

def generate_partition(seed, size, start_id):
    np.random.seed(seed)
    ids = np.arange(start_id, start_id + size, dtype=np.int64)
    base = datetime.utcnow()
    offsets = np.random.randint(0, 90*24*60*60, size=size)
    timestamps = [base - timedelta(seconds=int(o)) for o in offsets]
    categories = np.random.choice(["A","B","C","D","E","F"], size=size, p=[0.25,0.2,0.2,0.15,0.1,0.1])
    values = np.round(np.random.exponential(scale=50, size=size) + np.random.normal(0,5,size=size), 3)
    flags = np.random.choice([0,1], size=size, p=[0.85,0.15])
    df = pd.DataFrame({
        "id": ids,
        "timestamp": pd.to_datetime(timestamps),
        "category": categories,
        "value": values,
        "flag": flags
    })
    return df

def write_parquet_dataset(n_rows, n_parts):
    os.makedirs(PARQUET_DIR, exist_ok=True)
    chunk = n_rows // n_parts
    t0 = time.time()
    for part in range(n_parts):
        size = chunk if part < n_parts-1 else (n_rows - chunk*(n_parts-1))
        df_part = generate_partition(seed=part+123, size=size, start_id=part*chunk)
        file_path = os.path.join(PARQUET_DIR, f"part-{part:04d}.parquet")
        fastparquet.write(file_path, df_part, compression='SNAPPY')
        if (part+1) % 10 == 0 or part == n_parts-1:
            print(f"  wrote partition {part+1}/{n_parts} -> {file_path}")
    print("Write time (s):", time.time()-t0)

def run_analysis():
    print("Reading parquet with Dask...")
    ddf = dd.read_parquet(PARQUET_DIR, engine="pyarrow")
    print("npartitions:", ddf.npartitions, "| columns:", ddf.columns.tolist())

    # 1) per-category summary
    t1 = time.time()
    with ProgressBar():
        cat_summary = ddf.groupby("category").value.agg(["count","mean","std","sum"]).compute()
    t2 = time.time()
    print("cat_summary (time s):", round(t2-t1,2))
    print(cat_summary)

    # 2) daily timeseries (resample by day)
    t3 = time.time()
    with ProgressBar():
        ddf2 = ddf.set_index("timestamp")
        daily = ddf2.value.resample("1D").sum().compute()
    t4 = time.time()
    print("daily resample (time s):", round(t4-t3,2))
    print(daily.dropna().tail(10))

    # 3) filtered aggregation (flag==1)
    t5 = time.time()
    with ProgressBar():
        top_flag = (ddf[ddf.flag==1].groupby("category").value.mean().nlargest(5)).compute()
    t6 = time.time()
    print("flagged mean by category (time s):", round(t6-t5,2))
    print(top_flag)

    report = {
        "generated_rows": int(N_ROWS),
        "n_partitions": int(ddf.npartitions),
        "timings": {
            "cat_summary_s": round(t2-t1,2),
            "daily_resample_s": round(t4-t3,2),
            "flagged_s": round(t6-t5,2)
        }
    }
    with open(os.path.join(OUT_DIR, "report_summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Report written to", os.path.join(OUT_DIR, "report_summary.json"))

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Start: writing dataset to", PARQUET_DIR)
    write_parquet_dataset(N_ROWS, N_PARTITIONS)
    print("Dataset written — now analyzing with Dask")
    run_analysis()