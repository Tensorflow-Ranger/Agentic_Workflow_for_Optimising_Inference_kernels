import pyarrow as pa
from huggingface_hub import hf_hub_download
import pandas as pd

# Download the Arrow file from HuggingFace Hub
print("Loading ConCuR dataset from HuggingFace...")
arrow_file = hf_hub_download(
    repo_id='lkongam/ConCuR',
    filename='data-00000-of-00001.arrow',
    repo_type='dataset'
)

# Load using streaming format (the file is in Arrow streaming format, not IPC)
print(f"Loading Arrow file (streaming format)...\n")
with open(arrow_file, 'rb') as f:
    reader = pa.ipc.open_stream(f)
    table = reader.read_all()

df = table.to_pandas()

# Display dataset information
print("=" * 70)
print("CONCUR DATASET")
print("=" * 70)
print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print(f"\nColumn names and types:")
for col in df.columns:
    print(f"  {col:30} {str(df[col].dtype):15}")

# Display sample data
print(f"\n{'='*70}")
print("FIRST SAMPLE:")
print(f"{'='*70}")
for col in df.columns[:5]:
    val = df[col].iloc[0]
    if isinstance(val, str):
        if len(val) > 300:
            print(f"\n{col}:")
            print(f"  {val[:300]}...")
        else:
            print(f"\n{col}:")
            print(f"  {val}")
    else:
        print(f"\n{col}: {val}")

# Export the top 300 rows to CSV
output_csv = "concur_top_300.csv"
df.head(300).to_csv(output_csv, index=False)
print(f"\nSaved top 300 rows to {output_csv}")