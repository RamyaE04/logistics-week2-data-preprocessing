# Week 2: Logistics Data Collection and Preprocessing

## Objective
Prepare logistics shipment data for analysis by inspecting its structure, checking data quality, and creating a cleaned dataset.

## Potential Data Fields
Shipment date, distance, weight, transportation mode, traffic level, weather, delivery priority, planned delivery time, actual delivery time, and transportation cost.

## Preprocessing Workflow
1. Load the CSV file with Pandas.
2. Inspect rows, columns, data types, and summary statistics.
3. Check missing values and duplicate records.
4. Validate numerical values and categorical labels.
5. Convert date columns to suitable date/time types where needed.
6. Apply appropriate cleaning steps and document assumptions.
7. Save the cleaned data to a new CSV file.

## Example Inspection Code
```python
import pandas as pd

df = pd.read_csv("logistics_shipments_raw.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
```

## Files
- `week2.docx` — Week 2 report.
- `logistics_shipments_raw.csv` — raw dataset, if included.
- `logistics_shipments_clean.csv` — cleaned dataset, if included.
- Preprocessing code, if included.
- `requirements.txt` — dependencies.

## Data Note
The supplied demonstration uses simulated data, not records from a real logistics company. Describe only the cleaning steps actually performed.

## Install Dependencies
```bash
python -m pip install -r requirements.txt
```
