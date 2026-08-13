import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

file_path = r"Plant_1_Generation_Data.csv"

df = pd.read_csv(file_path)
print(df)

num_cols = [
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD"
]

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

print(test_df.shape)
print(train_df.shape)

print("\nTraining Data before scaling:")
print(train_df[num_cols].head())

print("\nTest Data before scaling:")
print(test_df[num_cols].head())

scaler = StandardScaler()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

print("Original dataset shape:", df.shape)

print("\nScaled Training Data:")
print(train_df[num_cols].head())

print("\nScaled Test Data:")
print(test_df[num_cols].head())