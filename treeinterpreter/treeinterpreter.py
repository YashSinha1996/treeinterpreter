# -*- coding: utf-8 -*-
import numpy as np
import sklearn
from scipy.sparse import csr_matrix

from sklearn.ensemble.forest import ForestClassifier, ForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from distutils.version import LooseVersion
from joblib import Parallel, delayed
from tqdm import tqdm

if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("treeinterpreter requires scikit-learn 0.17 or later")


def _get_tree_paths(tree, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]
    return paths


# def _predict_tree(model, X, joint_contribution=False, no_bias=False):
#     """
#     For a given DecisionTreeRegressor, DecisionTreeClassifier,
#     ExtraTreeRegressor, or ExtraTreeClassifier,
#     returns a triple of [prediction, bias and feature_contributions], such
#     that prediction ≈ bias + feature_contributions.
#     """
#     leaves = model.apply(X)
#     paths = _get_tree_paths(model.tree_, 0)
#
#     for path in paths:
#         path.reverse()
#
#     leaf_to_path = {}
#     # map leaves to paths
#     for path in paths:
#         leaf_to_path[path[-1]] = path
#
#         # remove the single-dimensional inner arrays
#     values = model.tree_.value.squeeze(axis=1)
#     # reshape if squeezed into a single float
#     if len(values.shape) == 0:
#         values = np.array([values])
#     if isinstance(model, DecisionTreeRegressor):
#         biases = np.full(X.shape[0], values[paths[0][0]])
#         line_shape = X.shape[1]
#     elif isinstance(model, DecisionTreeClassifier):
#         # scikit stores category counts, we turn them into probabilities
#         normalizer = values.sum(axis=1)[:, np.newaxis]
#         normalizer[normalizer == 0.0] = 1.0
#         values /= normalizer
#         if no_bias:
#             biases = []
#         else:
#             biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
#
#         line_shape = (X.shape[1], model.n_classes_)
#     direct_prediction = values[leaves]
#
#     # make into python list, accessing values will be faster
#     values_list = list(values)
#     feature_index = list(model.tree_.feature)
#
#     contributions = []
#     if joint_contribution:
#         for row, leaf in enumerate(leaves):
#             path = leaf_to_path[leaf]
#
#             path_features = set()
#             contributions.append({})
#             for i in range(len(path) - 1):
#                 path_features.add(feature_index[path[i]])
#                 contrib = values_list[path[i + 1]] - \
#                           values_list[path[i]]
#                 # path_features.sort()
#                 contributions[row][tuple(sorted(path_features))] = \
#                     contributions[row].get(tuple(sorted(path_features)), 0) + contrib
#         return direct_prediction, biases, contributions
#
#     else:
#         unique_leaves = np.unique(leaves)
#         unique_contributions = {}
#
#         for row, leaf in enumerate(unique_leaves):
#             for path in paths:
#                 if leaf == path[-1]:
#                     break
#
#             contribs = csr_matrix(line_shape)
#             for i in range(len(path) - 1):
#                 contrib = values_list[path[i + 1]] - \
#                           values_list[path[i]]
#                 contribs[feature_index[path[i]]] += contrib
#             unique_contributions[leaf] = contribs
#
#         for row, leaf in enumerate(leaves):
#             contributions.append(unique_contributions[leaf])
#
#         return direct_prediction, biases, np.mean(contributions, axis=0)

def _get_tree_contribs(values, feature_index, path, shape_req):
    contribs = csr_matrix(shape_req)
    for i in range(len(path) - 1):
        contrib = values[path[i + 1]] - \
                  values[path[i]]
        contribs[feature_index[path[i]]] += contrib
    return contribs


def _predict_tree(model, X):
    """
    For a given DecisionTreeRegressor, DecisionTreeClassifier,
    ExtraTreeRegressor, or ExtraTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    leaves = model.apply(X)
    paths = _get_tree_paths(model.tree_, 0)

    for path in paths:
        path.reverse()

    leaf_to_path = {}
    # map leaves to paths
    for path in paths:
        leaf_to_path[path[-1]] = path

        # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze(axis=1)
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])

    if isinstance(model, DecisionTreeClassifier):
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer
        # if no_bias:
        #     biases = []
        # else:
        #     biases = np.tile(values[paths[0][0]], (X.shape[0], 1))
    else:
        raise TypeError("Model is not a decision tree classifier")

    line_shape = (X.shape[1], model.n_classes_)
    direct_prediction = values[leaves]

    # make into python list, accessing values will be faster
    values_list = values
    feature_index = model.tree_.feature

    unique_leaves = np.unique(leaves)

    print(unique_leaves.shape, len(leaves))

    contribs_total = Parallel(n_jobs=180)(delayed(_get_tree_contribs)
                                                           (values_list, feature_index, leaf_to_path[leaf], line_shape)
                                                           for leaf in unique_leaves)
    # for row, leaf in enumerate(unique_leaves):
    #     path = leaf_to_path[leaf]
    #
    #     contribs = csr_matrix(line_shape)
    #     for i in range(len(path) - 1):
    #         contrib = values_list[path[i + 1]] - \
    #                   values_list[path[i]]
    #         contribs[feature_index[path[i]]] += contrib
    #     return contribs

    avg_contrib = sum(contribs_total)/len(contribs_total)
    # return direct_prediction, biases, contributions
    return avg_contrib


def _predict_forest(model, X):
    """
    For a given RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, or ExtraTreesClassifier returns a triple of
    [prediction, bias and feature_contributions], such that prediction ≈ bias +
    feature_contributions.
    """
    biases = []
    predictions = []

    num_trees = len(model.estimators_)
    line_shape = (X.shape[1], model.n_classes_)
    contributions = csr_matrix(line_shape)
    for tree in model.estimators_:
        contribution = _predict_tree(tree, X)
        contributions += contribution
        print(contributions, contribution, num_trees)
        input()

    contributions = contributions / num_trees

    # return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
    #         np.mean(contributions, axis=0))
    return predictions, biases, contributions


def predict(model, X):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.
    Parameters
    ----------
    model : DecisionTreeRegressor, DecisionTreeClassifier,
        ExtraTreeRegressor, ExtraTreeClassifier,
        RandomForestRegressor, RandomForestClassifier,
        ExtraTreesRegressor, ExtraTreesClassifier
    Scikit-learn model on which the prediction should be decomposed.

    X : array-like, shape = (n_samples, n_features)
    Test samples.
    
    joint_contribution : boolean
    Specifies if contributions are given individually from each feature,
    or jointly over them

    Returns
    -------
    decomposed prediction : triple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, If joint_contribution is False then returns and  array of 
        shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification, denoting
        contribution from each feature.
        If joint_contribution is True, then shape is array of size n_samples,
        where each array element is a dict from a tuple of feature indices to
        to a value denoting the contribution from that feature tuple.
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if isinstance(model, DecisionTreeClassifier):
        return _predict_tree(model, X)
    elif isinstance(model, ForestClassifier):
        return _predict_forest(model, X)
    else:
        raise ValueError("Wrong model type. Base learner needs to be a "
                         "DecisionTreeClassifier or DecisionTreeRegressor.")
