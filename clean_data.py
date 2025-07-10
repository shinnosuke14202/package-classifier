import pandas as pd

# Load raw data
raw = pd.read_csv("./data/raw_dataset.csv",
                  sep=';',  # Use semicolon instead of comma
                  encoding='ISO-8859-1',  # or try 'cp1252'
                  on_bad_lines='skip',  # skips lines with column mismatch
                  engine='python'  # more tolerant parser
                  )

# Select columns
filtered = raw[['appname', 'company', 'categorygame']]

# Convert all values to 'game' unless it's exactly 'no game'
filtered['categorygame'] = filtered['categorygame'].apply(
    lambda x: 'game' if x.strip().lower() != 'no game' else 'no game')

# Save to new CSV
filtered.to_csv("./data/filtered_dataset.csv", index=False)
