import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.weightstats import ztest

# Load Data
df = pd.read_csv(r"C:/Users/LENOVO/Downloads/Global_Music_Streaming_Listener_Preferences.csv")

# Basic info
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Descriptive statistics for numerical columns
print("\nDescriptive Stats:\n", df.describe())

# Unique values in categorical columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\nUnique values in {col}: {df[col].nunique()}")

# Set Seaborn style
sns.set(style="whitegrid")

# Histogram: Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram: Minutes Streamed Per Day
plt.figure(figsize=(10, 6))
sns.histplot(df['Minutes Streamed Per Day'], bins=30, kde=True, color='salmon')
plt.title('Daily Streaming Time Distribution')
plt.xlabel('Minutes per Day')
plt.ylabel('Users')
plt.show()

# Box Plot: Repeat Song Rate (%) by Subscription Type
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='Subscription Type',
    y='Repeat Song Rate (%)',
    hue='Subscription Type',
    data=df,
    palette='pastel',
    dodge=False,
    legend=False
)
plt.title('Repeat Song Rate by Subscription Type')
plt.show()

# Scatter Plot: Discover Weekly Engagement vs Minutes Streamed
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Discover Weekly Engagement (%)', y='Minutes Streamed Per Day', hue='Subscription Type', data=df)
plt.title('Engagement vs Streaming Time')
plt.xlabel('Discover Weekly Engagement (%)')
plt.ylabel('Minutes Streamed Per Day')
plt.show()

# Count Plot: Listening Time (Morning/Afternoon/Night)
plt.figure(figsize=(8, 6))
sns.countplot(
    x='Listening Time (Morning/Afternoon/Night)',
    hue='Listening Time (Morning/Afternoon/Night)',
    data=df,
    palette='Set2',
    legend=False
)
plt.title('Preferred Listening Time of Day')
plt.show()

# Correlation Heatmap
num_df = df.select_dtypes(include=['int64', 'float64'])
corr = num_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Z-Test: Premium vs Free users
premium = df[df['Subscription Type'] == 'Premium']['Minutes Streamed Per Day']
free = df[df['Subscription Type'] == 'Free']['Minutes Streamed Per Day']
z_stat, p_val = ztest(premium, free, alternative='larger')
print("Z-statistic:", z_stat)
print("P-value:", p_val)
if p_val < 0.05:
    print("✅ Reject Null Hypothesis: Premium users stream significantly more.")
else:
    print("❌ Fail to Reject Null Hypothesis: No significant difference.")

# Country-Based Streaming Behavior
country_minutes = df.groupby('Country')['Minutes Streamed Per Day'].mean().sort_values(ascending=False).head(10)
country_df = country_minutes.reset_index()
country_df.columns = ['Country', 'AvgMinutes']
plt.figure(figsize=(12, 6))
sns.barplot(
    data=country_df,
    y='Country',
    x='AvgMinutes',
    hue='Country',
    palette='coolwarm',
    dodge=False,
    legend=False
)
plt.title('Top 10 Countries by Avg. Daily Streaming Minutes')
plt.xlabel('Average Minutes Streamed')
plt.ylabel('Country')
plt.show()

# Listening Time of Day – Streaming Patterns
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Listening Time (Morning/Afternoon/Night)',
    y='Minutes Streamed Per Day',
    data=df,
    palette='Set3',
    hue='Listening Time (Morning/Afternoon/Night)',
    legend=False
)
plt.title('Streaming Time by Listening Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Minutes Streamed')
plt.show()
