"""
TITLE: DECISION TREE
NAME: NGUYEN HUU DAC
JVN - ICT
11/2018
"""
import copy
import csv
import math
import os
import time
from random import randrange

from graphviz import Digraph

dirname = os.path.dirname(__file__)
os.environ["PATH"] += os.pathsep + os.path.join(dirname, 'Graphviz2.38/bin')


# Load a CSV file
def load_csv(file_name):
    file = open(file_name)
    lines = csv.DictReader(file, delimiter=',')
    data = list(lines)
    return data


# save to csv
def save_csv(data, file_path):
    file = open(file_path, mode='w')
    fieldnames = data[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    fieldnames = list(fieldnames)
    writer.writeheader()
    for line in data:
        writer.writerow({fieldnames[0]: line[fieldnames[0]], fieldnames[1]: line[fieldnames[1]]})
    file.close()
    return data


# Decision tree class
class DecisionTree:
    def __init__(self, node_name, ig, predictor=None, properties=None):
        self.node = node_name
        self.ig = ig
        self.properties = properties
        self.predictor = predictor  # hard predictor

    def set_properties(self, properties):
        self.properties = properties

    def get_node_name(self):
        return self.node

    def get_information_gain(self):
        return self.ig

    def check_predictor(self, property_name):
        if property_name not in self.properties:
            return 2
        if 'predictor' in self.properties[property_name].keys():
            return 0
        return 1

    def get_predictor(self, property_name):
        return self.properties[property_name]['predictor']

    def get_properties(self):
        return self.properties

    def get_next_node(self, property_name):
        return self.properties[property_name]['next']

    def get_predictor_hard(self):
        return self.predictor


def select_predictor(train_data, attributes):
    predictor = ''
    predictor_count = {}
    for line in train_data:
        if line[attributes[0]] not in predictor_count.keys():
            predictor_count[line[attributes[0]]] = 1
        else:
            predictor_count[line[attributes[0]]] += 1
    highest_prob = 0
    for class_value in predictor_count:
        if predictor_count[class_value] > highest_prob:
            highest_prob = predictor_count[class_value]
            predictor = class_value
    return {"predictor": predictor, "highest_prob": highest_prob}


def check_early_stopping(train_data, attributes, current_depth, max_depth):
    result_predictor = select_predictor(train_data, attributes)
    predictor = result_predictor['predictor']
    highest_prob = result_predictor['highest_prob']
    # Get predictor
    if current_depth == max_depth or highest_prob / len(train_data) >= 0.85:
        # choose predictor
        return {"code": 1000, "predictor": predictor}
    return {"code": 0, "predictor": predictor}


# Calculate entropy of each property
def entropy_property(list_property, total):
    entropy = 0
    for property_class in list_property:
        p = list_property[property_class] / total
        entropy -= p * math.log2(p)

    return entropy


# Calculate Gini for each property of an attribute
def gini_property(list_property, total):
    """
    gini_of_property
    property = {"count": ..., "class": {...}}
    :rtype: float
    """
    gini = 1
    for property_class in list_property:
        p = list_property[property_class] / total
        gini -= p * p

    return gini


def entropy_class_attribute(train_data, main_class):
    total = len(train_data)
    # count all properties of attribute that need to be classified
    list_property = {}
    for line in train_data:
        if line[main_class] in list_property.keys():
            list_property[line[main_class]] += 1
        else:
            list_property[line[main_class]] = 1
    # Entropy or Gini
    entropy = gini_property(list_property, total)

    return entropy


def entropy_attribute(train_data, decision, attribute_name):
    total = len(train_data)

    # get subset of each attribute
    list_property = {}
    for line in train_data:
        if line[attribute_name] in list_property.keys():
            list_property[line[attribute_name]]["count"] += 1  # count whole data of this property
        else:
            list_property[line[attribute_name]] = {}
            list_property[line[attribute_name]]["count"] = 1
            list_property[line[attribute_name]]["class"] = {}

            # count class of this property
        if line[decision] in list_property[line[attribute_name]]["class"].keys():
            list_property[line[attribute_name]]["class"][line[decision]] += 1
        else:
            list_property[line[attribute_name]]["class"][line[decision]] = 1

    entropy_of_attribute = 0
    for property in list_property:
        entropy_of_attribute += list_property[property]["count"] / total * gini_property(
            list_property[property]['class'],
            list_property[property]["count"])
    return entropy_of_attribute


# def choose decision node & new dataset
def select_decision_node(train_data, main_class, attributes):
    # Get entropy of attribute that need to classify
    entropy_class = entropy_class_attribute(train_data, main_class)

    # Calculate entropy of each attributes
    list_entropy = {}
    for attribute in attributes:
        list_entropy[attribute] = entropy_attribute(train_data, main_class, attribute)

    # find largest information gain and select decision node
    largest_ig = 0
    decision_node = None
    for attribute in list_entropy:
        ig = entropy_class - list_entropy[attribute]  # information gain

        if ig >= largest_ig:
            largest_ig = ig
            decision_node = attribute

    # split data based on new decision node
    new_data = {}
    list_property = {}
    for line in train_data:
        if line[decision_node] not in new_data.keys():
            new_data[line[decision_node]] = []  # create new dataset
            list_property[line[decision_node]] = 1  # Get list property and count
        list_property[line[decision_node]] += 1  # count
        new_data[line.pop(decision_node)].append(line)  # pop out that attribute

    result = {"code": 0, "node": decision_node, "ig": largest_ig, "properties": list_property,
              "dataset": new_data}
    return result


def select_decision_node_recursive(train_data, current_depth, max_depth):
    # Get attribute and calculate information gain
    attributes = list(train_data[0].keys())
    # Check early stopping
    result_check_early_stopping = check_early_stopping(train_data, attributes, current_depth, max_depth)
    if result_check_early_stopping['code'] == 1000:
        return result_check_early_stopping

    # Select node for next decision
    result_decision_node = select_decision_node(train_data, attributes[0], attributes[1:])
    node = DecisionTree(node_name=result_decision_node['node'],
                        ig=result_decision_node['ig'],
                        predictor=result_check_early_stopping["predictor"])

    decision = {}
    for property in result_decision_node['dataset']:
        # decision: {'property':{num: , next:{node: }}, 'property':{}}
        decision[property] = {}
        decision[property]["num"] = len(result_decision_node['dataset'][property])
        # recursive decision node
        next = select_decision_node_recursive(result_decision_node['dataset'][property],
                                              current_depth + 1,
                                              max_depth)

        # Stop loop and choose predictor
        if next['code'] == 0:
            decision[property]["next"] = next['estimator']
        else:
            decision[property]["predictor"] = next['predictor']

    node.set_properties(decision)
    return {"code": 0, "estimator": node}


def train_decision_tree_model(train_data, max_depth):
    # check max_depth
    if max_depth >= (len(train_data[0].keys()) - 2):
        max_depth = len(train_data[0].keys()) - 2
    model = select_decision_node_recursive(train_data, 0, max_depth)
    return model['estimator']


# print tree recursive
def print_tree_recursive(f, current_node, current_node_name):
    for branch in current_node.get_properties():
        if current_node.check_predictor(branch) == 0:
            predictor = current_node.get_predictor(branch)
            predictor_name = predictor + current_node_name + branch

            f.attr('node', shape='diamond')
            f.node(predictor_name, label=predictor)
            f.edge(current_node_name, predictor_name, label=branch)
            f.attr('node', shape='circle')
        else:
            next_node = current_node.get_next_node(branch)
            next_node_label = next_node.get_node_name()
            next_node_name = next_node_label + current_node_name + branch

            f.node(next_node_name, label=next_node_label)
            f.edge(current_node_name, next_node_name, label=branch)
            print_tree_recursive(f, next_node, next_node_name)
    return True


# print
def print_tree(decision_tree_model: DecisionTree, file_name):
    # {node: , info: {information gain: , decision: {'property':{num: , predictor: or next:{node: }}, 'property':{}}}
    f = Digraph('decision_tree', filename=file_name)
    f.attr(rankdir='LR', size='8,5')

    f.attr('node', shape='circle')
    f.node(decision_tree_model.get_node_name())
    print_tree_recursive(f, decision_tree_model, decision_tree_model.get_node_name())
    f.view()


# recursive predict function
def predict_recursive(current_node: DecisionTree, line):
    # Check first attribute (node)
    property_data = line[current_node.get_node_name()]
    property_model = current_node.get_properties()

    check_predictor = current_node.check_predictor(property_data)
    if check_predictor == 0:
        # Get predictor of this tuple
        predictor = current_node.get_predictor(property_data)
    elif check_predictor == 2:
        predictor = current_node.get_predictor_hard()
    else:
        next_node = current_node.get_next_node(property_data)
        # Recursive to next branch
        predictor = predict_recursive(next_node, line)
    return predictor


# Execute predict function
def predict(current_node, test_data):
    attributes = list(test_data[0].keys())  # Get class attribute
    decision_attribute = attributes[1]

    result = []
    count = 0  # Count correct answer
    total = 0  # Count total tuple
    # Loop all tuple in a table
    for line in test_data:
        # Check first attribute (node)
        predicted_value = predict_recursive(current_node, line)
        total += 1
        if predicted_value == line[decision_attribute]:
            count += 1
        list_line = {attributes[0]: line[attributes[0]], attributes[1]: predicted_value}
        result.append(list_line)
    accurate = round(count / total, 2)
    return {"correct": count, "total": total, "accurate": accurate, "list": result}


# Split a dataset into k folds
def cross_validation_split(data, folds=10):
    dataset_split = list()
    dataset_copy = list(data)
    fold_size = int(len(data) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# run train_decision_tree_model with k-fold
def train_decision_tree_k_folds(data, max_depth=4, k_folds=10):
    data_list = cross_validation_split(data, k_folds)
    current = 0
    model_list = []
    # loop all k_folds
    while current < k_folds:
        # prepare data for train model
        temp_data = copy.deepcopy(data_list)
        test_data = temp_data.pop(current)
        train_data = []
        for fold in temp_data:
            for line in fold:
                line.pop(list(line.keys())[0])
                train_data.append(line)

        model = train_decision_tree_model(train_data, max_depth=max_depth)
        # validate
        result = predict(model, test_data)
        model = {"model": model, "accurate": result['accurate']}
        model_list.append(model)
        # print("Fold ", current, ": Done with accuracy about: ", result['accurate'])
        current += 1

    highest_accurate = 0.0  # Highest accuracy model
    mean_accuracy = 0  # mean value
    for model in model_list:
        if model['accurate'] >= highest_accurate:
            highest_accurate = model['accurate']
        mean_accuracy += model['accurate']
    mean_accuracy = mean_accuracy / len(model_list)
    return {"mean_acc": mean_accuracy}


def main(path):
    train_data = load_csv(os.path.join(path, 'train_data.csv'))
    test_data = load_csv(os.path.join(path, 'test_data.csv'))

    print("Training decision model with k-folds...")
    highest_acc = 0
    best_depth = 0
    for depth in range(3, 7):
        # print('\nDepth = ', depth)
        result_fold = train_decision_tree_k_folds(train_data, max_depth=depth, k_folds=10)
        if result_fold['mean_acc'] > highest_acc:
            highest_acc = result_fold['mean_acc']
            best_depth = depth

    # Use best depth to train tree
    # pop-out all id in dataset
    model_data = []
    for line in train_data:
        line.pop(list(line.keys())[0])
        model_data.append(line)
    model = train_decision_tree_model(model_data, max_depth=best_depth)

    print("We choose best method depth = ", best_depth, " with accuracy = ", highest_acc, "\nPredicting...")
    result = predict(model, test_data)
    print("Result of predicting test data")
    print("Total correct: ", result['correct'], " Over: ", result['total'])
    print("Accuracy: ", result["accurate"])

    save_csv(result['list'], os.path.join(path, 'result.csv'))  # save to file
    print_tree(model, os.path.join(path, 'graph.gv'))


path = os.path.join(dirname, 'data')
list_folder = os.listdir(path)
for folder in list_folder:
    print('\nProcessing dataset:', folder)
    start_time = time.time()
    main(os.path.join(path, folder))
    print('Running time: ', time.time() - start_time)
