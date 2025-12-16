
import pandas as pd
from io import BytesIO

def extract_tables(file_bytes: bytes, ext: str):
    if ext == "csv":
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))

    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(map(str, row.values)))

    return rows
