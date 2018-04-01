import matplotlib.pyplot as plt
import numpy as np

def load_emnist(dataset_name):
    print('loading test data')
    with open("{}-test.csv".format(dataset_name), "r") as test_file:
        test_matrix = np.loadtxt(test_file, delimiter=',', dtype=int)
        test_labels = test_matrix[:, 0]
        test_data = test_matrix[:, 1:]

    print('loading training data')        
    with open("{}-train.csv".format(dataset_name), "r") as train_file:
        train_matrix = np.loadtxt(train_file, delimiter=',', dtype=int)
        train_labels = train_matrix[:, 0]
        train_data = train_matrix[:, 1:]

    def filter_labels(labels, data):
        relevant_labels = list(range(10, 36))
        mask = np.isin(labels, relevant_labels)
        return labels[mask], data[mask]

    def reshape_data(data):
        l = len(data)
        d =  data.reshape((l, 28, 28))
        return np.transpose(d, (0, 2, 1))

    filtered_test_labels, filtered_test_data = filter_labels(test_labels, test_data)
    filtered_train_labels, filtered_train_data = filter_labels(train_labels, train_data)

    return filtered_test_labels, reshape_data(filtered_test_data), filtered_train_labels, reshape_data(filtered_train_data)

test_labels, test_data, train_labels, train_data = load_emnist('emnist-byclass')
test = np.concatenate((test_labels.reshape((len(test_labels), 1)), test_data.reshape((len(test_data), 784))), axis=1)
train = np.concatenate((train_labels.reshape((len(train_labels), 1)), train_data.reshape((len(train_data), 784))), axis=1)

np.savetxt('test.csv', test, delimiter=',')
np.savetxt('train.csv', test, delimiter=',')
# def display(img, threshold=0.5):
#     # Debugging only
#     render = ''
#     for row in img:
#         for col in row:
#             if col > threshold:
#                 render += '@'
#             else:
#                 render += '.'
#         render += '\n'
#     return render

# def find_labels(labels, data):
#     for i in range(10, 62):
#         index = np.where(labels == i)[0][0]
#         print(i)
#         plt.axis('off')
#         plt.imshow(data[index], cmap='gray')
#         plt.show()

print('test_labels shape = {}'.format(test_labels.shape))
print('test_data shape = {}'.format(test_data.shape))
print('train_labels shape = {}'.format(train_labels.shape))
print('train_data shape = {}'.format(train_data.shape))

# tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
# batch_size = 32
# with tf.device('/cpu:0'):
#     real_images = train_data[:batch_size]
#         # Sanity check that we're getting images.
#     check_real_digits = tfgan.eval.image_reshaper(
#         real_images[:20,...], num_cols=10)
#     visualize_digits(check_real_digits)
