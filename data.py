import os 
import pickle
import numpy as np
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf

class Data:
    def __init__(self) -> None:
        
        # TODO: Fill this in based on where you saved the training and testing data
        folder_path = './traffic-signs-data'
        training_file = os.path.join(folder_path, 'train.p')
        validation_file= os.path.join(folder_path, 'valid.p')
        testing_file = os.path.join(folder_path, 'test.p')
        self.num_sign_types = 43 # number of sign classes 

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']

        # store original data for comparison
        self.orig_y_train = [y for y in self.y_train]
        self.orig_y_valid = [y for y in self.y_valid]
        self.orig_y_test = [y for y in self.y_test]

        # combine training, validation, and test features and labels
        self.X_comb = np.vstack((self.X_train, self.X_valid, self.X_test))
        self.y_comb = np.hstack((self.y_train, self.y_valid, self.y_test))

        # categorize all image data into one of 43 sign types
        self.img_id_dict = {i:[] for i in range(self.num_sign_types)}
        for img, sign_id in zip(self.X_comb, self.y_comb):
            self.img_id_dict[sign_id].append(img)
        

        # initialize variables to store split training, validation, and test data
        split_X_train = []
        split_y_train = []
        split_X_valid = []
        split_y_valid = []
        split_X_test = [] 
        split_y_test = [] 

        # all the data unsplit    
        X_all = []
        y_all = []

        # within each category, do a split into train(70%), valid(10%), test (20%)
        percent_train, percent_valid, percent_test = 0.8, 0.1, 0.2
        for sign_id, images in self.img_id_dict.items():
            # all images of class with id equal sign_id
            X = images
            # list equal to number of images with value equal to sign_id 
            y = np.full((len(X)), sign_id)
            # shuffle images to randomize (all y so no need to shuffle)
            X = shuffle(X) 
            n_train = int(len(X) * percent_train)
            n_valid = int(len(X) * percent_valid)
            n_test = int(len(X) * percent_test)
            # n_test = int(len(X) * percent_test) not needed since indices taken from the end of train+valid to end of list
            X_train, y_train = X[:n_train], y[:n_train] 
            X_valid, y_valid = X[n_train: n_train + n_valid], y[n_train: n_train + n_valid]
            X_test, y_test = X[-n_test:], y[-n_test:]

            # set split data equal to dataset when empty but np.vstack when not to add data 
            if split_X_train == []:
                split_X_train = X_train
            else:
                split_X_train = np.vstack((split_X_train, X_train))
            
            if split_y_train == []:
                split_y_train = y_train
            else:
                split_y_train = np.hstack((split_y_train, y_train))
            
            if split_X_valid == []:
                split_X_valid = X_valid
            else:
                split_X_valid = np.vstack((split_X_valid, X_valid))
            
            if split_y_valid == []:
                split_y_valid = y_valid
            else:
                split_y_valid = np.hstack((split_y_valid, y_valid))
            
            if split_X_test == []:
                split_X_test = X_test
            else:
                split_X_test = np.vstack((split_X_test, X_test)) 
            
            if split_y_test == []:
                split_y_test = y_test
            else:
                split_y_test = np.hstack((split_y_test, y_test))

            # all the data unsplit
            if X_all == []:
                X_all = X
            else:
                X_all = np.vstack((X_all, X))
            
            if y_all == []:
                y_all = y
            else:
                y_all = np.hstack((y_all, y))
            # print("X all shape ", X_all.shape)
            # print("y all shape ", y_all.shape)



        self.X_train, self.y_train = split_X_train, split_y_train
        self.X_valid, self.y_valid = split_X_valid, split_y_valid
        self.X_test, self.y_test = split_X_test, split_y_test




    def get_train_data(self):
        return (self.X_train, self.y_train)

    def get_valid_data(self):
        return (self.X_valid, self.y_valid)

    def get_test_data(self):
        """ 
        gets test data 
        Return
        ------
        (X_test, y_t)
        """
        return (self.X_test, self.y_test)

    def split_train_valid_data(self, X_data, y_data, split_percent=0.9):
        """ 
        splits training and validation data using split_percent 
        
        Return
        ------
        splits data into training and validation set (X_train, y_train, X_valid, y_valid)

        """
        
        shuffled_X, shuffled_y = shuffle(X_data, y_data, random_state=0)
        n = int(len(X_data) * split_percent)
        X_train, y_train = shuffled_X[:n], shuffled_y[:n]
        X_valid, y_valid = shuffled_X[n:], shuffled_y[n:]
        return (X_train, y_train, X_valid, y_valid)

    def brighten_and_contrast_image(self, image):
        image = np.asarray(image).astype('float32')
        # image = tf.io.decode_image(image, channels=channels)
        # image = tf.convert_to_tensor(image)
        # convert from int to float for calculations            
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # adjust contrast and brightness
        if tf.math.reduce_mean(image) < 0.1:
            image = tf.image.adjust_contrast(image, 3)
            image = tf.image.adjust_brightness(image, 0.2)
            
        return image

    def dim_and_decontrast_image(self, image):
        channels = 3
        image = np.asarray(image).astype('float32')
        # image = tf.io.decode_image(image, channels=channels)
        # image = tf.convert_to_tensor(image)
        # convert from int to float for calculations            
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        # adjust contrast and brightness
        if tf.math.reduce_mean(image) > 0.1:
            image = tf.image.adjust_contrast(image, 0.5)
            image = tf.image.adjust_brightness(image, -0.2)
            
        return image

    def augment_data(self, X_data, y_data):
        """ 
        accepts feature and label data and returns augmented data that 
        1) normalizes original data 2) brightens and increases contrast 
        of normalized data 3) dims and reduces contrast of original data
        """
        X_normalized = []
        X_brightened_and_contrasted = []
        X_dimmed_and_decontrasted = []
        y_normalized = []
        y_brightened_and_contrasted = []
        y_dimmed_and_decontrasted = []        
        for i in range(len(X_data)):
            img = self.X_train[i]
            y = y_data[i]
            img = np.array(img, dtype=np.float32)
            norm = (img - 168) / 168
            X_normalized.append(norm)
            y_normalized.append(y)

            bright_contrast = self.brighten_and_contrast_image(norm)
            X_brightened_and_contrasted.append(bright_contrast)
            y_brightened_and_contrasted.append(y)

            dim_decontrast = self.dim_and_decontrast_image(norm)
            X_dimmed_and_decontrasted.append(dim_decontrast)
            y_dimmed_and_decontrasted.append(y)

        aug_X = np.vstack((X_normalized, X_brightened_and_contrasted, X_dimmed_and_decontrasted))        
        aug_y = np.hstack((y_normalized, y_brightened_and_contrasted, y_dimmed_and_decontrasted)) 

        return (aug_X, aug_y)

    def get_aug_data(self):
        return (self.aug_X_train, self.aug_y_train) #, self.aug_X_valid, self.aug_y_valid)        


    def count_sign_type(self, ids):
        id_counts = {id:0 for id in range(self.num_sign_types)}
        for id in ids:
            id_counts[id] += 1

        sign_count = []
        # store count in index numerical order (ascending)
        for id in range(self.num_sign_types):

            sign_count.append((id, id_counts[id]))
        return sign_count
        
    def get_rebalanced_train_valid_test_counts(self):
        _, y_train = self.get_train_data()
        _, y_valid = self.get_valid_data()
        _, y_test = self.get_test_data()

        train_counts = self.count_sign_type(y_train)
        valid_counts = self.count_sign_type(y_valid)
        test_counts = self.count_sign_type(y_test)

        # extract count at respective index
        train = [count for _, count in train_counts]
        valid = [count for _, count in valid_counts]
        test = [count for _, count in test_counts]        
        return (train, valid, test)

    def get_original_train_valid_test_counts(self):

        train_counts = self.count_sign_type(self.orig_y_train)
        valid_counts = self.count_sign_type(self.orig_y_valid)
        test_counts = self.count_sign_type(self.orig_y_test)

        # extract count at respective index
        train = [count for _, count in train_counts]
        valid = [count for _, count in valid_counts]
        test = [count for _, count in test_counts]        
        return (train, valid, test)