#  ____        _ _ _     ____        _                  _   
# / ___| _ __ | (_) |_  |  _ \  __ _| |_ __ _ ___  __ _| |_ 
# \___ \| '_ \| | | __| | | | |/ _` | __/ _` / __|/ _` | __|
#  ___) | |_) | | | |_  | |_| | (_| | || (_| \__ \ (_| | |_ 
# |____/| .__/|_|_|\__| |____/ \__,_|\__\__,_|___/\__,_|\__|
#       |_|                                                 
#
# This script splits the dataset.json file into train.json and test.json
# with 17 queries per command in train and 3 queries per command in test

import json
import os
import random
from collections import defaultdict

# Define the directory and file paths
current_directory = os.path.dirname(os.path.realpath(__file__))
input_file_path = os.path.join(current_directory, "dataset.json")
train_file_path = os.path.join(current_directory, "train.json")
test_file_path = os.path.join(current_directory, "test.json")

# Initialize data structures
command_to_queries = defaultdict(list)
training_data = []
testing_data = []

# Load the dataset as a JSON array
with open(input_file_path, "r", encoding="utf-8") as file:
    dataset = json.load(file)
    # Group the queries by command
    # This will allow us to split the data by command
    # commands are the keys and queries are the values
    for item in dataset:
        command = item["command"]
        command_to_queries[command].append(item)

# Split data: 17 for training, 3 for testing per command
for command, items in command_to_queries.items():
    # Shuffle to ensure random selection
    random.shuffle(items)
    
    # Get train and test samples
    if len(items) >= 20:
        train_samples = items[:17]
        test_samples = items[17:20]
    else:
        train_count = int(len(items) * 0.85)
        train_samples = items[:train_count]
        test_samples = items[train_count:]
    
    # Add to respective datasets
    training_data.extend(train_samples)
    testing_data.extend(test_samples)

# Shuffle the final datasets
random.shuffle(training_data)
random.shuffle(testing_data)

# Save the train dataset in JSONL format
with open(train_file_path, "w", encoding="utf-8") as file:
    for item in training_data:
        json.dump(item, file)
        file.write("\n")

# Save the test dataset in JSONL format
with open(test_file_path, "w", encoding="utf-8") as file:
    for item in testing_data:
        json.dump(item, file)
        file.write("\n")

print(f"Split complete!")
print(f"Train data: {len(training_data)} samples")
print(f"Test data: {len(testing_data)} samples")
print(f"Number of unique commands: {len(command_to_queries)}")
