import cozmo
import datetime
import numpy as np
import pickle
import random
import re
import sys
import time

from cozmo.util import degrees, distance_mm, speed_mmps
from skimage import io, feature, filters, exposure, color
from sklearn import svm, metrics


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
        # extract feature vector from image data
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


def run(sdk_conn):
    # Basic Setup
    print("Initializing")
    robot = sdk_conn.wait_for_robot()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    print("----------------------------------------------------------")

    # Prepare Image Classifier
    myarg = sys.argv[1:]
    if len(myarg) > 0 and "train" in myarg:
        img_clf = train()
    else:
        try:
            img_clf = pickle.load(open('model.sav', 'rb'))
            print('Model loaded from: ''model.sav''')
        except:
            img_clf = train()
    print("----------------------------------------------------------")

    # Go to idle state
    idle(robot, img_clf)


def train():
    print('Training in progress')
    startTime = time.time()
    img_clf = ImageClassifier()
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    train_data = img_clf.extract_image_features(train_raw)
    img_clf.train_classifier(train_data, train_labels)
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')
    test_data = img_clf.extract_image_features(test_raw)
    img_clf.train_classifier(test_data, test_labels)

    pickle.dump(img_clf, open('model.sav', 'wb'))
    endTime = time.time()
    print('Training complete')
    print('Training time: ', str(datetime.timedelta(seconds=(endTime - startTime))))
    print('Model saved as: ''model.sav''')

    return img_clf


def idle(robot, img_clf):
    count = 0
    while True:
        robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        data = []
        time.sleep(0.5)

        if count < 10:
            pass
        elif count%4 == 0:
            robot.turn_in_place(degrees(50)).wait_for_completed()
            robot.drive_straight(distance_mm(random.uniform(-20,20)), speed_mmps(80)).wait_for_completed()
        else:
            robot.turn_in_place(degrees(-10)).wait_for_completed()

        count += 1

        latestImage = robot.world.latest_image
        newImage = latestImage.raw_image
        data.append(np.array(newImage))
        feature = img_clf.extract_image_features(data)

        symbol = img_clf.classifier.predict(feature)
        if symbol == "drone":
            drone(robot)
        elif symbol == "order":
            order(robot)
            count = 0
        elif symbol == "inspection":
            inspection(robot)
            count = 0

def drone(robot):
    robot.say_text('drone').wait_for_completed()
    time.sleep(0.5)
    lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
    cubes = robot.world.wait_until_observe_num_objects(num=1, object_type=cozmo.objects.LightCube, timeout=60)
    lookaround.stop()

    if len(cubes) > 0:
        robot.pickup_object(cubes[0], num_retries=5).wait_for_completed()
        robot.drive_straight(distance_mm(100), speed_mmps(80)).wait_for_completed()
        robot.place_object_on_ground_here(cubes[0]).wait_for_completed()
        robot.drive_straight(distance_mm(-80), speed_mmps(80)).wait_for_completed()

def order(robot):
    robot.say_text('order').wait_for_completed()
    time.sleep(0.5)
    robot.drive_wheels(100, 40, duration=9.9)

def inspection(robot):
    robot.say_text('inspection').wait_for_completed()
    time.sleep(0.5)
    for x in range(4):
        robot.set_lift_height(1, accel=.5, duration=2)
        robot.set_lift_height(0, accel=.5, duration=2, in_parallel=True, num_retries=225)
        robot.drive_straight(distance_mm(200), speed_mmps(50), in_parallel=True).wait_for_completed()
        robot.turn_in_place(degrees(90)).wait_for_completed()
    robot.set_lift_height(0, accel=.5).wait_for_completed()

if __name__ == '__main__':
    cozmo.setup_basic_logging()

    try:
        cozmo.connect(run)
    except cozmo.ConnectionError as e:
        sys.exit("A connection error occurred: %s" % e)
