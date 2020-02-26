import numpy as np
import re
import pickle
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color


class ImageClassifier:

    def __init__(self):
        self.classifier = None

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
        self.classifier = svm.LinearSVC().fit(train_data, train_labels)


    def predict_labels(self, data):
        # Predict labels of test data using trained model in self.classifier
        predicted_labels = self.classifier.predict(data)
        return predicted_labels


def main():
    img_clf = ImageClassifier()

    # Load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')

    # Convert images into features
    train_data = img_clf.extract_image_features(train_raw)

    # Train model
    img_clf.train_classifier(train_data, train_labels)

    pickle.dump(img_clf, open('model.sav', 'wb'))
    img_clf2 = pickle.load(open('model.sav', 'rb'))

if __name__ == "__main__":
    main()
