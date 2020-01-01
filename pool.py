from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skmultiflow.trees import HoeffdingTree
import matplotlib.pyplot as plt

res = []
# Create a dataset.
X, y = make_classification(10000, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Define a tree for fitting the complete dataset and one for streaming.
ht_complete = HoeffdingTree()
ht_partial = HoeffdingTree()

# Fit the complete dataset.
ht_complete.fit(X_train, y_train)
ht_complete_score = ht_complete.score(X_test, y_test)
print(f'Score when fitting at once: {ht_complete_score}')

# Streaming samples one after another.
timer = False
j = 0
for i in range(len(X_train)):
    ht_partial.partial_fit(X_train[i].reshape(1, -1), np.array([y_train[i]]))
    res.append(ht_partial.score(X_test, y_test))
    print(f'Score when streaming after {i} samples: {res[-1]}')
    if res[-1] >= ht_complete_score - 0.01:
        print(f'(Almost) full score reached! Continue for another {20 - j} samples.')
        timer = True
    if timer:
        j += 1
        if j == 20:
            break

# Plot the scores after each sample.
plt.figure(figsize=(12, 6))
plt.plot([0, i], [ht_complete_score, ht_complete_score], '--', label='Hoeffding Tree built at once')
plt.plot(res, label='Incrementally built Hoeffding Tree')
plt.xlabel('Number of Samples', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.title('Fitting a Hoeffding Tree at once (enough Memory available) vs fitting it via Streaming', fontsize=20)
plt.legend()
