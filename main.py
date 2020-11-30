import cv2
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import json
import os
import pickle
from pathlib import Path
from typing import Tuple
from sklearn.svm import SVC


# TODO: versions of libraries that will be used:
#  Python 3.9 (you can use previous versions as well)
#  numpy 1.19.4
#  scikit-learn 0.22.2.post1
#  opencv-python 4.2.0.34


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(data: np.ndarray, feature_detector_descriptor, vocab_model) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def data_processing(x: np.ndarray) -> np.ndarray:
    # TODO: add data processing here
    return x


def project():
    np.random.seed(42)

    # TODO: fill the following values
    first_name = 'Michał'
    last_name = 'Gontarczyk'

    data_path = Path('drive/My Drive/train')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    images, labels = load_dataset(data_path)

    train_images, test_images, train_labes, test_labels = train_test_split(images, labels, train_size=0.85,
                                                                           random_state=42, stratify=labels)
    # TODO: create a detector/descriptor here. Eg. cv2.AKAZE_create()
    feature_detector_descriptor = cv2.BRISK_create()
    train_descriptors = []

    for image_tr in train_images:
        for descriptor in feature_detector_descriptor.detectAndCompute(image_tr, None)[1]:
            train_descriptors.append(descriptor)

    # train_descriptors = [descriptor for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1] for image in train_images]

    print(len(train_descriptors))
    NB_WORDS = 256

    kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42)
    kmeans.fit(train_descriptors)

    X_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans)
    y_train = train_labes

    X_test = apply_feature_transform(test_images, feature_detector_descriptor, kmeans)
    y_test = test_labels

    classifier = RandomForestClassifier(n_estimators=600, min_samples_split=10, criterion='gini', random_state=42,
                                        verbose=20, max_features='log2', min_samples_leaf=1)
    classifier.fit(X_train, y_train)

    classifier_1 = GradientBoostingClassifier(n_estimators=500, max_depth=100, random_state=42)
    classifier_1.fit(X_train, y_train)

    classifier_2 = SVC(C=0.1, verbose=20, random_state=42, kernel='poly')
    classifier_2.fit(X_train, y_train)

    classifier_4 = SVC(verbose=20, random_state=42, kernel='poly')
    classifier_4.fit(X_train, y_train)

    # TODO: train a vocabulary model and save it using pickle.dump function
    # with Path('drive/My Drive/vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
    #     vocab_model = pickle.load(vocab_file)
    #
    x_transformed = apply_feature_transform(train_images, feature_detector_descriptor, kmeans)
    #
    # TODO: train a classifier and save it using pickle.dump function
    # with Path('drive/My Drive/clf.p').open('rb') as classifier_file:  # Don't change the path here
    #     clf = pickle.load(classifier_file)

    score = classifier.score(x_transformed, labels)
    print(f'{first_name} {last_name} score: {score}')

    score = classifier_1.score(x_transformed, labels)
    print(f'{first_name} {last_name} score: {score}')

    score = classifier_2.score(x_transformed, labels)
    print(f'{first_name} {last_name} score: {score}')

    score = classifier_4.score(x_transformed, labels)
    print(f'{first_name} {last_name} score: {score}')


if __name__ == '__main__':
    project()