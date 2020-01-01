from sklearn.ensemble import RandomTreesEmbedding
from sklearn.datasets import load_digits
from sklearn.tree.export import export_text
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Decoder class.
class RandomTreesEmbeddingDecoder:
    def __init__(self, random_trees_embedding):
        self.random_trees_embedding = random_trees_embedding
        self.constraints = self._get_tree_constraints()

    def encode(self, X):
        output = self.random_trees_embedding.transform(X)
        indices = []
        start = 0
        for estimator in self.random_trees_embedding.estimators_:
            n_leaves = estimator.get_n_leaves()
            indices.append(output[:, start:start + n_leaves].argmax(axis=1))
            start = start + n_leaves
        return np.array(indices).transpose()

    def decode(self, X_encoded, dim, method='minimum'):
        res_ = []
        for x in X_encoded:
            res = dim * [0]
            constraints = [self.constraints[i][x[i]] for i in
                           range(self.random_trees_embedding.get_params()['n_estimators'])]
            constraints = self._simplify_constraints(sum(constraints, []))
            if method == 'minimum':
                for row in constraints:
                    if row[1] != -np.inf:
                        res[int(row[0])] = row[1]
                    else:
                        res[int(row[0])] = row[2]
            elif method == 'maximum':
                for row in constraints:
                    if row[2] != np.inf:
                        res[int(row[0])] = row[2]
                    else:
                        res[int(row[0])] = row[1]
            elif method == 'mean':
                for row in constraints:
                    if row[1] != -np.inf and row[2] != np.inf:
                        res[int(row[0])] = (row[1] + row[2]) / 2.
                    else:
                        if row[1] != -np.inf:
                            res[int(row[0])] = row[1]
                        else:
                            res[int(row[0])] = row[2]
            res_.append(res[:])
        return np.array(res_)

    @staticmethod
    def _parse_line(l):
        res = []
        for s in l:
            feature_name, direction, bound = s.split()[-3:]
            feature_name = int(feature_name.split('_')[-1])
            if direction == '<=':
                res.append((feature_name, -np.inf, float(bound)))
            elif direction == '>':
                res.append((feature_name, float(bound), np.inf))
            else:
                raise ValueError
        return res

    @staticmethod
    def _simplify_constraints(constraints):
        constraints_df = pd.DataFrame(
            [(constraint[0], constraint[1], constraint[2]) for constraint in constraints],
            columns=['Variable Name', 'Lower Bound', 'Upper Bound']
        )
        return constraints_df.groupby('Variable Name', as_index=False).agg(
            {'Lower Bound': 'max', 'Upper Bound': 'min'}).values

    def _get_tree_constraints(self):
        res_ = []
        for t in self.random_trees_embedding.estimators_:
            res = []
            constraints = []
            leaf_depth = 0
            visited_leaf = False

            for line in export_text(t, max_depth=np.inf).split('\n')[:-1]:
                if '<=' in line or '>' in line:
                    if visited_leaf:
                        for _ in range(leaf_depth - line.count('|   ')):
                            constraints.pop()
                        visited_leaf = False
                    constraints.append(line)
                else:
                    res.append(self._parse_line(constraints[:]))
                    visited_leaf = True
                    leaf_depth = line.count('|   ')
            res_.append(res[:])
        return res_


# Load data.
data = load_digits()
X = data.data

# Fit the encoder and define the decoder.
rte = RandomTreesEmbedding(n_estimators=20, sparse_output=False, max_depth=50)
rte.fit(X)
rted = RandomTreesEmbeddingDecoder(rte)

# Encode and decode the pictures.
e = rted.encode(X)
d = rted.decode(e, dim=64, method='mean')
np.abs(X - d).mean()

# Plot a single number.
to_plot = 0
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.imshow(X[to_plot].reshape(8, 8))
ax1.set_title('Original')

ax3 = fig.add_subplot(122)
ax3.imshow(d[to_plot].reshape(8, 8))
ax3.set_title('RFE Reconstruction')
