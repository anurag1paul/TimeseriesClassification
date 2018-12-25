from data_loader import DataLoader
from preprocessing import prepare_data


# data file
data_file = "challenge_dataset.xlsx"

# train and test files
train_file = "train.csv"
test_file = "test.csv"

# test size
test_size = 0.2

# prepare train and test datasets
prepare_data(data_file, train_file, test_file, test_size)

# define dataset loader
data_loader = DataLoader(train_file, test_file)

# Train and evaluate models

# Comparison Plot
