"""
generate_sample_data.py
-----------------------
Generates a synthetic sample dataset that mirrors the schema of the
"Vehicle Dataset from CarDekho" (Car details v3.csv) so the project can be
cloned and run end-to-end without needing to download the original dataset.

For best results, replace data/Car_details_v3.csv with the real dataset from:
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

Run:
    python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

brands = {
    "Maruti Swift": (60000, 900000),
    "Hyundai i20": (70000, 950000),
    "Honda City": (150000, 1400000),
    "Toyota Innova": (300000, 2200000),
    "Ford Ecosport": (150000, 1300000),
    "Mahindra XUV500": (200000, 1800000),
    "Tata Nexon": (150000, 1300000),
    "Skoda Octavia": (250000, 2200000),
    "BMW 3 Series": (900000, 4500000),
    "Audi A4": (900000, 4500000),
}
names = list(brands.keys())

fuels = ["Diesel", "Petrol", "CNG", "LPG"]
fuel_p = [0.45, 0.48, 0.05, 0.02]
seller_types = ["Individual", "Dealer", "Trustmark Dealer"]
seller_p = [0.6, 0.32, 0.08]
transmissions = ["Manual", "Automatic"]
trans_p = [0.85, 0.15]
owners = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
owner_p = [0.55, 0.28, 0.12, 0.05]

rows = []
for _ in range(N):
    name = np.random.choice(names)

    year = int(np.random.randint(2005, 2024))
    age = 2026 - year

    km_driven = int(max(500, np.random.normal(60000, 35000) + age * 4000))
    fuel = np.random.choice(fuels, p=fuel_p)
    seller_type = np.random.choice(seller_types, p=seller_p)
    transmission = np.random.choice(transmissions, p=trans_p)
    owner = np.random.choice(owners, p=owner_p)

    # engine/power scale roughly with the segment implied by the brand's base range,
    # so that engine & power (which stay in the model) carry real predictive signal.
    base_low, _ = brands[name]
    segment = np.clip((base_low - 60000) / (900000 - 60000), 0, 1)  # 0=hatchback .. 1=premium

    mileage = round(np.clip(np.random.normal(20 - 6 * segment, 3), 8, 32), 2)
    engine = int(np.clip(np.random.normal(900 + segment * 1800, 200), 700, 3000))
    max_power = round(np.clip(np.random.normal(60 + segment * 200, 15), 35, 300), 2)
    seats = int(np.random.choice([4, 5, 5, 5, 7, 7, 8], p=[0.05, 0.55, 0.0, 0.0, 0.25, 0.10, 0.05]))
    torque = f"{int(np.clip(np.random.normal(150, 60), 60, 500))}Nm@ {np.random.randint(1500,4000)}rpm"

    # Price is driven mainly by engine size & power (the features the model can see),
    # plus age depreciation, km wear, transmission, and ownership history.
    base_price = 40000 + engine * 220 + max_power * 3200

    depreciation = np.exp(-0.09 * age)
    km_penalty = np.exp(-km_driven / 400000)
    owner_penalty = {"First Owner": 1.0, "Second Owner": 0.9,
                      "Third Owner": 0.8, "Fourth & Above Owner": 0.7}[owner]
    trans_boost = 1.08 if transmission == "Automatic" else 1.0

    price = base_price * depreciation * km_penalty * owner_penalty * trans_boost
    price *= np.random.normal(1.0, 0.05)
    price = max(30000, price)

    rows.append([
        name, year, round(price, -2), km_driven, fuel, seller_type,
        transmission, owner, f"{mileage} kmpl", f"{engine} CC",
        f"{max_power} bhp", torque, seats
    ])

df = pd.DataFrame(rows, columns=[
    "name", "year", "selling_price", "km_driven", "fuel", "seller_type",
    "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"
])

out_path = "data/Car_details_v3.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
