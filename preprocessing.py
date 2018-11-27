# Load a CSV file
import csv
import os
from random import randrange


def load_csv(file_name):
    file = open(file_name)
    lines = csv.DictReader(file, delimiter=',')
    data = list(lines)
    return data


def save_csv(data, file_path):
    file = open(file_path, mode='w')
    fieldnames = data[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for line in data:
        row = {}
        for attribute in fieldnames:
            row[attribute] = line[attribute]
        writer.writerow(row)
    file.close()
    return data


def split_to_train_test_file(data):
    size_data_test = len(data) * 1 / 4
    test_data = []
    while len(test_data) < size_data_test:
        index = randrange(len(data))
        test_data.append(data.pop(index))
    return {"train_data": data, "test_data": test_data}


dirname = os.path.dirname(__file__)
dataset = split_to_train_test_file(load_csv(os.path.join(dirname, 'data/car/car.csv')))
train_data = dataset['train_data']
test_data = dataset['test_data']
save_csv(train_data, os.path.join(dirname, 'data/car/train_data.csv'))
save_csv(test_data, os.path.join(dirname, 'data/car/test_data.csv'))

