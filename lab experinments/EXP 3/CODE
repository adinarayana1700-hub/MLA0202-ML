# EXPERIMENT 3: ID3 DECISION TREE

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],

    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],

    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],

    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],

    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes',
             'No', 'Yes', 'No', 'Yes', 'Yes',
             'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("Dataset:")
display(df)

# Convert categorical values to numerical values
X = pd.get_dummies(df.drop('Play', axis=1))

y = df['Play']

# Create ID3 Decision Tree
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(X, y)

print("\nDecision Tree Accuracy:")
print(model.score(X, y))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------

new_sample = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['High'],
    'Wind': ['Strong']
})

new_sample_encoded = pd.get_dummies(new_sample)

new_sample_encoded = new_sample_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

prediction = model.predict(new_sample_encoded)

print("\nNew Sample:")
display(new_sample)

print("Predicted Class:", prediction[0])


# -----------------------------
# DECISION TREE GRAPH
# -----------------------------

plt.figure(figsize=(18, 10))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)

plt.title("ID3 Decision Tree")

plt.show()
