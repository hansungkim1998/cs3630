#!/usr/bin/env python

import numpy as np
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color


class ImageClassifier:
    def __init__(self):
        self.classifer = None

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # Read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)

        # Create one large array of image data
        data = io.concatenate_images(ic)

        # Extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data,labels)

    def extract_image_features(self, data):
        # Extract feature vector from image data
        imgList = []

        for img in data:
            modifiedImg = filters.gaussian(img, sigma = 0.8, multichannel = False)
            fVector = feature.hog(modifiedImg, orientations = 12,
                            pixels_per_cell = (27,27), cells_per_block = (4,4),
                            feature_vector = True, block_norm = 'L2-Hys')
            imgList.append(fVector)

        feature_data = np.array(imgList)
        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # Train model and save the trained model to self.classifier
        self.classifer = svm.LinearSVC().fit(train_data, train_labels)

    def predict_labels(self, data):
        # Predict labels of test data using trained model in self.classifier
        predicted_labels = self.classifer.predict(data)
        return predicted_labels


def main():
    img_clf = ImageClassifier()

    # Load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # Convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # Train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    # Test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n",metrics.confusion_matrix(test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
