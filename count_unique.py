import pandas as pd

# Đọc file CSV
df = pd.read_csv('vocabulary (1).csv')

# Đếm số hàng khác nhau trong cột Vertical1
unique_count = df['Name'].nunique()

print(f"Số hàng khác nhau trong cột Vertical1: {unique_count}")
