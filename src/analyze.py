import pandas as pd
import numpy as np

# Open the CSV files
male_df = pd.read_csv('data/male_data.csv')
female_df = pd.read_csv('data/female_data.csv')
train_df = pd.read_csv('data/train.csv')

print("=" * 80)
print("MALE DATA ANALYSIS")
print("=" * 80)
print("\nDataFrame Info:")
print(male_df.info())
print("\nDataFrame Shape:", male_df.shape)
print("\nFirst few rows:")
print(male_df.head())

# Calculate statistics for male data
print("\n--- MALE DATA STATISTICS ---")
male_stats = pd.DataFrame({
    'Mean': male_df.mean(numeric_only=True),
    'Min': male_df.min(numeric_only=True),
    'Max': male_df.max(numeric_only=True)
})
print(male_stats)

print("\n" + "=" * 80)
print("FEMALE DATA ANALYSIS")
print("=" * 80)
print("\nDataFrame Info:")
print(female_df.info())
print("\nDataFrame Shape:", female_df.shape)
print("\nFirst few rows:")
print(female_df.head())

# Calculate statistics for female data
print("\n--- FEMALE DATA STATISTICS ---")
female_stats = pd.DataFrame({
    'Mean': female_df.mean(numeric_only=True),
    'Min': female_df.min(numeric_only=True),
    'Max': female_df.max(numeric_only=True)
})
print(female_stats)

print("\n" + "=" * 80)
print("TRAIN DATA ANALYSIS")
print("=" * 80)
print("\nDataFrame Info:")
print(train_df.info())
print("\nDataFrame Shape:", train_df.shape)
print("\nFirst few rows:")
print(train_df.head())

# Calculate statistics for train data
print("\n--- TRAIN DATA STATISTICS ---")
train_stats = pd.DataFrame({
    'Mean': train_df.mean(numeric_only=True),
    'Min': train_df.min(numeric_only=True),
    'Max': train_df.max(numeric_only=True)
})
print(train_stats)

print("\n" + "=" * 80)
print("COMBINED STATISTICS COMPARISON")
print("=" * 80)

# Create a combined view if the columns match
if set(male_df.columns) == set(female_df.columns) == set(train_df.columns):
    print("\n--- COMPARISON ACROSS ALL DATASETS ---")
    for col in male_df.select_dtypes(include=[np.number]).columns:
        print(f"\nParameter: {col}")
        comparison = pd.DataFrame({
            'Male_Mean': [male_df[col].mean()],
            'Male_Min': [male_df[col].min()],
            'Male_Max': [male_df[col].max()],
            'Female_Mean': [female_df[col].mean()],
            'Female_Min': [female_df[col].min()],
            'Female_Max': [female_df[col].max()],
            'Train_Mean': [train_df[col].mean()],
            'Train_Min': [train_df[col].min()],
            'Train_Max': [train_df[col].max()]
        })
        print(comparison.to_string(index=False))
else:
    print("Note: Column structures differ between datasets")