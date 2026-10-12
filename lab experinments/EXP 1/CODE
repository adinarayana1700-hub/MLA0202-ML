# EXPERIMENT 1: FIND-S ALGORITHM

import pandas as pd
import matplotlib.pyplot as plt

# Training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Training Data:")
display(df)

# FIND-S Algorithm
attributes = list(df.columns[:-1])

# Start with most specific hypothesis
hypothesis = ['0'] * len(attributes)

print("\nInitial Hypothesis:")
print(hypothesis)

# Process each positive example
for index, row in df.iterrows():

    if row['EnjoySport'] == 'Yes':

        for i in range(len(attributes)):

            if hypothesis[i] == '0':
                hypothesis[i] = row[attributes[i]]

            elif hypothesis[i] != row[attributes[i]]:
                hypothesis[i] = '?'

        print("After example", index + 1, ":", hypothesis)

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)

# -----------------------------
# GRAPH
# -----------------------------

# Convert hypothesis into numerical values for visualization
values = []

for h in hypothesis:
    if h == '?':
        values.append(2)
    elif h == '0':
        values.append(0)
    else:
        values.append(1)

plt.figure(figsize=(10, 5))

plt.bar(attributes, values)

plt.title("FIND-S Final Hypothesis")
plt.xlabel("Attributes")
plt.ylabel("Hypothesis State")

plt.yticks(
    [0, 1, 2],
    ['Specific (0)', 'Specific Value', 'General (?)']
)

plt.xticks(rotation=45)

plt.show()
