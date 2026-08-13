import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

file_path = r"Plant_1_Generation_Data.csv"

df = pd.read_csv(file_path)
print(df)

feature = "DC_POWER"

print("Original statistics:")
print(df[feature].describe())

# Calculate Q1 and Q3
q1 = df[feature].quantile(0.25)
q3 = df[feature].quantile(0.75)

# Calculate IQR
IQR = q3 - q1

# Calculate lower and upper fences
lower_fence = q1 - 1.5 * IQR
upper_fence = q3 + 1.5 * IQR

print("\nQ1 =", q1)
print("Q3 =", q3)
print("Lower fence =", lower_fence)
print("Upper fence =", upper_fence)

# Identify outliers
outliers = df[
    (df[feature] < lower_fence) |
    (df[feature] > upper_fence)
]

print("\nNumber of outliers:", len(outliers))

# Clip the outliers
df["DC_POWER_Clipped"] = df[feature].clip(
    lower=lower_fence,
    upper=upper_fence
)

print("\nMinimum BEFORE clipping:")
print(df[feature].min())

print("\nMaximum BEFORE clipping:")
print(df[feature].max())

print("\nMinimum AFTER clipping:")
print(df["DC_POWER_Clipped"].min())

print("\nMaximum AFTER clipping:")
print(df["DC_POWER_Clipped"].max())

# Train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

print("\nTraining shape:", train_df.shape)
print("Testing shape:", test_df.shape)

# MinMax Scaling
scaler = MinMaxScaler()

train_df["DC_POWER_Scaled"] = scaler.fit_transform(
    train_df[["DC_POWER_Clipped"]]
)

test_df["DC_POWER_Scaled"] = scaler.transform(
    test_df[["DC_POWER_Clipped"]]
)

print("\nTraining Data:")
print(
    train_df[
        ["DC_POWER", "DC_POWER_Clipped",
         "DC_POWER_Scaled"]
    ].head()
)

print("\nTest Data:")
print(
    test_df[
        ["DC_POWER", "DC_POWER_Clipped",
         "DC_POWER_Scaled"]
    ].head()
)

print("\nOriginal dataset shape:", df.shape)