"""
This script contains common functions for data handling and evaluation.
"""
import numpy as np
import json
import csv
import os.path


def readJSONFile(path):
    """
    Returns the content of JSON file at path
    """
    print(f"\nREADING JSON FILE: {path}")
    with open(path, encoding='utf8') as json_file:
        data = json.load(json_file)
    if isinstance(data, list):
        print(f"FOUND {len(data)} ITEMS")
    return data


def writeCSV(file_path, data, ignore=[]):
    """
    Writes data, a list of dicts, to a CSV file at file_path
    """
    exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        fields = [key for key in data[0].keys() if key not in ignore]
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        if not exists:            
            w.writeheader()
        w.writerows(data)


def mean(l):
    """
    Returns the mean of a list of numbers l
    """
    if not len(l):
        return np.nan
    return sum(l)/float(len(l))
