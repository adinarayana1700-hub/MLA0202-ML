# EXPERIMENT 2: CANDIDATE-ELIMINATION ALGORITHM

import pandas as pd
import matplotlib.pyplot as plt

# Training dataset
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

attributes = list(df.columns[:-1])

# Initial boundaries
specific_h = ['0'] * len(attributes)
general_h = [['?'] * len(attributes)]

print("\nInitial Specific Boundary:")
print(specific_h)

print("\nInitial General Boundary:")
print(general_h)


# Candidate Elimination
for index, row in df.iterrows():

    example = [row[attr] for attr in attributes]

    if row['EnjoySport'] == 'Yes':

        # Generalize specific hypothesis
        for i in range(len(attributes)):

            if specific_h[i] == '0':
                specific_h[i] = example[i]

            elif specific_h[i] != example[i]:
                specific_h[i] = '?'

    else:
        # Negative example
        for i in range(len(attributes)):

            if specific_h[i] != example[i]:
                general_h[0][i] = specific_h[i]

    print("\nAfter Example", index + 1)
    print("Specific Boundary:", specific_h)
    print("General Boundary:", general_h)


print("\n==============================")
print("FINAL VERSION SPACE")
print("==============================")

print("Specific Boundary:")
print(specific_h)

print("\nGeneral Boundary:")
print(general_h)


# -----------------------------
# GRAPH
# -----------------------------

specific_values = []

for value in specific_h:

    if value == '?':
        specific_values.append(2)
    elif value == '0':
        specific_values.append(0)
    else:
        specific_values.append(1)


general_values = []

for value in general_h[0]:

    if value == '?':
        general_values.append(2)
    else:
        general_values.append(1)


plt.figure(figsize=(10, 5))

x = range(len(attributes))

plt.plot(x, specific_values, marker='o', label='Specific Boundary')
plt.plot(x, general_values, marker='s', label='General Boundary')

plt.xticks(x, attributes, rotation=45)

plt.yticks(
    [0, 1, 2],
    ['Initial', 'Specific Value', 'General (?)']
)

plt.title("Candidate-Elimination Version Space")
plt.xlabel("Attributes")
plt.ylabel("Hypothesis Generality")

plt.legend()
plt.grid(True)

plt.show()
