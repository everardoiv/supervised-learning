import csv
import pandas as pd


def write_headers(filename, header):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


def write_file(read_file, write_file):
    with open(read_file, newline='') as f:
        reader = csv.reader(f)
        append_data_to_file(write_file, reader)


def append_data_to_file(filename, data):
    with open(filename, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def merge_files(header, file1, file2, file_out):
    if header:
        write_headers(file_out, header)
    write_file(file1, file_out)
    write_file(file2, file_out)


def file_statistics(filename):
    data = pd.read_csv(filename)
    print(data.head(3))
    print(data.describe(include='all'))
    print(data.isnull().values.any())
    if filename.__contains__('poker'):
        print(data['Poker Hand'].value_counts())
    else:
        print(data['num'].value_counts())


def sample_data(filename, weights, version, frac, rs):
    data = pd.read_csv(filename)
    sampled_data = data.sample(frac=frac, replace=True, weights=weights, random_state=rs)
    file_out = 'data/output/sampled-' + version + '-' + filename[-19:]
    sampled_data.to_csv(file_out, index=None, header=True)

    return file_out


if __name__ == "__main__":
    # Poker Hand Dataset
    attribute_header = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Poker Hand']
    file_to_write = 'data/output/poker-hand-data.csv'
    file_1 = 'data/input/poker-hand-testing.data'
    file_2 = 'data/input/poker-hand-training-true.data'
    final_file = 'data/output/sampled-poker-hand-data.csv'

    # merge_files(attribute_header, file_1, file_2, file_to_write)
    # sample_file_1 = sample_data(file_to_write, None, '1', 0.02, 4)
    # sample_file_2 = sample_data(file_to_write, 'Poker Hand', '2', 0.08, 1)
    # Watch out it replicates the headers so easiest to manually delete
    # merge_files(None, sample_file_1, sample_file_2, final_file)
    file_statistics(final_file)

    # Heart Disease Dataset
    # heart_disease_header = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    #                         'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    # heart_disease_data_file = 'data/output/heart-disease-data.csv'
    # write_headers(heart_disease_data_file, heart_disease_header)
    # write_file('data/input/processed.cleveland.data', heart_disease_data_file)
    # file_statistics(heart_disease_data_file)

    # Credit Card Dataset
    # cc_data = pd.read_csv('data/input/credit-card-data.csv').drop(['ID'], axis=1)
    # print(cc_data.head())
    # print(cc_data.describe(include='all'))
    # print(cc_data.isnull().values.any())
    # print(cc_data['default payment next month'].value_counts())
    # cc_data.to_csv('data/output/credit-card-data.csv', index=None, header=True)
