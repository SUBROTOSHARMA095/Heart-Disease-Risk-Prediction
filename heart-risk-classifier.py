from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

train_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'ROC-AUC': []}
test_scores  = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'ROC-AUC': []}

# data (as pandas dataframes) 
X = copy.deepcopy(heart_disease.data.features)
y = heart_disease.data.targets 

no_of_samples=len(X)

y = (y['num'] > 0).astype(int)

for i in range(5):
    random_indices=np.random.permutation(no_of_samples)
    no_training_samples = int(no_of_samples*0.7)
    X_train = X.iloc[random_indices[:no_training_samples]].copy()
    y_train = y.iloc[random_indices[:no_training_samples]].copy()

    X_train['ca'] = X_train['ca'].fillna(X_train['ca'].median())
    X_train['thal'] = X_train['thal'].fillna(X_train['thal'].mode()[0])

    X_test = X.iloc[random_indices[no_training_samples:]].copy()
    y_test = y.iloc[random_indices[no_training_samples:]].copy()

    X_test['ca'] = X_test['ca'].fillna(X_test['ca'].median())
    X_test['thal'] = X_test['thal'].fillna(X_test['thal'].mode()[0])

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc = roc_auc_score(y_test,y_probs)

    test_scores['Accuracy'].append(accuracy)
    test_scores['F1'].append(f1)
    test_scores['Precision'].append(precision)
    test_scores['Recall'].append(recall)
    test_scores['ROC-AUC'].append(roc)

    train_scores['Accuracy'].append(accuracy_score(y_train, model.predict(X_train)))
    train_scores['F1'].append(f1_score(y_train, model.predict(X_train)))
    train_scores['Precision'].append(precision_score(y_train, model.predict(X_train)))
    train_scores['Recall'].append(recall_score(y_train, model.predict(X_train)))
    train_scores['ROC-AUC'].append(roc_auc_score(y_train, model.predict_proba(X_train)[:,1]))

plt.figure(figsize=(10, 5))
x = range(10)

for metric in train_scores.keys():
    combined = train_scores[metric] + test_scores[metric]
    plt.plot(x, combined, marker='o', linewidth=2, label=metric)

# Dividing line and light shading on the Test side
plt.axvline(x=4.5, color='black', linestyle='--', linewidth=1.5)
plt.axvspan(4.5, 9.5, color='gray', alpha=0.08)

# Labels for all 10 positions
x_labels = ['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5',
            'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
plt.xticks(x, x_labels, rotation=20)
plt.xlim(-0.5, 9.5)
plt.ylim(0.5, 1.02)
plt.ylabel('Score')
plt.title('Decision Tree Metrics Across 5 Random Runs (Train vs. Test)')
plt.legend(loc='lower left')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print('Training scores:')
for i,j in train_scores.items():
    print(i,":",j)
print()
print('Test scores: ')
for i,j in test_scores.items():
    print(i,':',j)
