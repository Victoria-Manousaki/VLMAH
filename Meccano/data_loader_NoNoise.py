import tensorflow as tf
import os
import numpy as np
import csv
import time
import threading
import gc
import random
import glob
from random import randrange
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import import_pre_extracted, calculate_timesteps

"""# **Data generator**"""


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""

    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))

    return gen


def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape

    if os.path.exists(image):
        resize = cv2.imread(image)

        dim = (w, h)
        # resize image
        image = cv2.resize(resize, dim, interpolation=cv2.INTER_AREA)
        img_arr = np.asarray(image)
    else:

        raise ValueError('Image path is invalid')

    # Turn it into numpy, normalize and return.
    x = (img_arr / 255.).astype(np.float32)

    return x


class DataSet():

    def __init__(self, image_shape=(224, 224, 3), seq_length=None, data_type=None, sampling_type=0, TrainValTest=None, model_name=None):
        self.seq_length = seq_length
        self.data_type = data_type
        self.sampling_type = sampling_type
        self.image_shape = image_shape
        self.TrainValTest = TrainValTest
        self.model_name = model_name

        # 1. Get the data
        self.data = self.get_data()

        # 2. Get the classes
        self.classes = self.get_classes()

    def get_words_ActionPrediction(self):
        tokenizer = Tokenizer()
        path = os.path.join('splits')

        xs_temp, labels_temp = [], []

        if self.TrainValTest == 'train':
            path = path + '/train_onlypairs_NoUnderscore.txt'

            data = open(path).read()
            corpus = data.lower().split("\n")
            tokenizer.fit_on_texts(corpus)


        else:
            # Load the tokenizer from the train
            path = path + '/train_onlypairs_NoUnderscore.txt'

            data = open(path).read()
            corpus = data.lower().split("\n")
            tokenizer.fit_on_texts(corpus)

            path = os.path.join('splits')
            if self.TrainValTest == 'val':
                path = path + '/val_onlypairs_NoUnderscore.txt'

            else:
                path = path + '/test_onlypairs_NoUnderscore.txt'


            data = open(path).read()
            corpus = data.lower().split("\n")

        total_words = len(tokenizer.word_index)
        print(tokenizer.word_index)

        count = 0
        idx_end_action = []
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                count += 1
                if (token_list[i] == 1 & (token_list[i - 1] != 33 or token_list[i - 1] != 34 or token_list[i + 1] != 33 or token_list[i + 1] != 34)):

                    # if i>1 :
                    idx_end_action.append(count)

                    n_gram_sequence = token_list[:i]
                    input_sequences.append(n_gram_sequence)
        # pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        # create predictors and label

        SeqSamp_Ids = []
        countAc = 0
        for i in range(len(input_sequences)):
            if (input_sequences[i, -2] == 0 or input_sequences[i, -1] == 0):
                pass

            else:

                SeqSamp_Ids.append([countAc, countAc + 1])
                # Verb
                xs_temp.append(input_sequences[i, :-2])

                labels_temp.append(input_sequences[i, -2])

                # Noun
                xs_temp.append(input_sequences[i, :-2])
                labels_temp.append(input_sequences[i, -1])


                countAc += 2

        xs = np.array(xs_temp)
        labels = np.array(labels_temp)

        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words + 1)

        return xs, ys, np.array(idx_end_action), total_words + 1, xs, tokenizer, SeqSamp_Ids

    def get_words_ActionPrediction_Action(self):
        tokenizer = Tokenizer()
        path = os.path.join('splits')

        xs_temp, labels_temp = [], []

        if self.TrainValTest == 'train':

            path = path + '/fullaction/train_allactions.txt'
            data = open(path).read()
            corpus = data.lower().split("\n")

            tokenizer.fit_on_texts(corpus)


        else:
            # Load the tokenizer from the train

            path = path + '/fullaction/train_allactions.txt'
            data = open(path).read()
            corpus = data.lower().split("\n")

            tokenizer.fit_on_texts(corpus)

            path = os.path.join('splits')
            if self.TrainValTest == 'val':

                path = path + '/fullaction/val_allactions.txt'
            else:

                path = path + '/fullaction/test_allactions.txt'

            data = open(path).read()
            corpus = data.lower().split("\n")

        total_words = len(tokenizer.word_index)
        print(tokenizer.word_index)

        count = 0
        idx_end_action = []
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]


            for i in range(2, len(token_list)):
                count += 1
                if (token_list[i] == 1 ):

                            idx_end_action.append(count)

                            n_gram_sequence = token_list[:i]
                            input_sequences.append(n_gram_sequence)


        # pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        # create predictors and label

        SeqSamp_Ids = []
        countAc = 0

        if len(self.data) != len(input_sequences):
            print(len(self.data))
            print(len(input_sequences))
            raise ValueError('I did not found the same sample number between .txt and .csv files')

        for i in range(len(input_sequences)):
            if (input_sequences[i, -2] == 0 or input_sequences[i, -1] == 0):
                pass

            else:

                SeqSamp_Ids.append([countAc, countAc + 1])

                # Action
                xs_temp.append(input_sequences[i, :-1])

                labels_temp.append(input_sequences[i, -1])
                if input_sequences[i, -1] == 53 or input_sequences[i, -1] == 51:
                    print(input_sequences[i])
                    raise ValueError('Should not have 1, 49, 51')

                countAc += 1

        xs = np.array(xs_temp)
        labels = np.array(labels_temp)

        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words + 1)

        return xs, ys, np.array(idx_end_action), total_words + 1, xs, tokenizer, SeqSamp_Ids


    # Current object prediction
    def get_words(self):
        tokenizer = Tokenizer()
        path = os.path.join('splits')

        if self.TrainValTest == 'train':
            path = path + '/train_onlypairs_NoUnderscore.txt'

            data = open(path).read()
            corpus = data.lower().split("\n")
            tokenizer.fit_on_texts(corpus)

        else:
            # Load the tokenizer from the train
            path = path + '/train_onlypairs_NoUnderscore.txt'

            data = open(path).read()
            corpus = data.lower().split("\n")
            tokenizer.fit_on_texts(corpus)

            path = '/splits'
            if self.TrainValTest == 'val':
                path = path + '/val_onlypairs_NoUnderscore.txt'
            else:
                path = path + '/test_onlypairs_NoUnderscore.txt'

            data = open(path).read()
            corpus = data.lower().split("\n")

        total_words = len(tokenizer.word_index)

        count = 0
        idx_end_action = []
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                count += 1
                if (token_list[i] == 1 & (
                        token_list[i - 1] != 33 or token_list[i - 1] != 34 or token_list[i + 1] != 33 or token_list[
                    i + 1] != 34)):
                    # if i>1 :
                    idx_end_action.append(count)

                    n_gram_sequence = token_list[:i]
                    input_sequences.append(n_gram_sequence)
        # pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        xs, labels, input_sequences_fix = [], [], []
        # create predictors and label
        for i in range(len(input_sequences)):
            if (input_sequences[i, -2] == 0 or input_sequences[i, -1] == 0):
                pass
            else:
                xs.append(input_sequences[i, :-1])
                labels.append(input_sequences[i, -1])

        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words + 1)

        return xs, ys, np.array(idx_end_action), total_words + 1, np.array(input_sequences_fix)

    def get_data(self):
        """Load our data from file."""
        path = os.path.join('splits')
        if self.TrainValTest == 'train':
            path = path + '/train.csv'
        elif self.TrainValTest == 'val':

            path = path + '/val.csv'
        else:
            path = path + '/test.csv'
        #

        with open(path, encoding='utf-8', mode='r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        # actor_id, class_id, class name, frame_start, frame_end
        return data

    def get_classes(self):
        """Extract the classes from our data."""
        classes = []

        for item in self.data:

            if item[2] not in classes:
                classes.append(item[2])

        # Sort them.
        classes = sorted(classes)

        # Return.
        return classes

    def get_frames_for_sample(self, sample, RandomClip=False, seq_len=8, data_type='images', feat_mod='obj', timesteps = True):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""

        # Get the id of actor

        filename = str(int(sample[0]))
        filename.encode('utf-8').decode('utf-8')

        path = r'/Mecanno/frames'

        # sample the segment
        frame_start = int(sample[3][:-4])
        frame_end = int(sample[4][:-4])
        all_images = sorted(glob.glob((path + '/' + str(filename) + '**/*.jpg')))

        sgm_back = 0
        if timesteps:

          frame_enda, sgm_back = calculate_timesteps(filename, frame_end, self.TrainValTest)
          if frame_enda[0] in  ['0']:
            frame_end = int(format(int(frame_enda), '04d'))
          else:
              frame_end = int(frame_enda)

          if frame_start == frame_end:
              frame_end = frame_end + 2
          elif frame_end<frame_start:

              frame_end = int(sample[4][:-4])
              frame_end = frame_end + 2



          images = all_images[frame_start - 1:frame_end - 1]


          #images = [path + '\\' + str(filename) + '\\' + img + '.jpg' for img in frames_paths]

          if images:
              pass
          else:
              print(frame_start)
              print(frame_end)
              raise ValueError('Did not find any images!!!')


          if data_type in ['images']:


              images_out = self.fix_ma_Imagesequence(images=images)

          elif data_type in ['features']:
              ###### Call import_pre_extracted ######
              if self.TrainValTest in ['train', 'retrain']:
                  mode_ext = 'Train_Val'
              else:
                  mode_ext = 'Test'

              images_out = import_pre_extracted(sample, images, feat_mod, mode_ext)

              if len(images_out) != self.seq_length:
                  images_out = self.fix_ma_sequence(temp_arr=images_out)
          else:
              raise ValueError('Unsupported data type, please type "images" or "features"')

        else:
          

            if frame_start == frame_end:
                frame_end = frame_end + 2
            images = all_images[frame_start - 1:frame_end - 1]



            if data_type in ['images']:
                if len(images) != 0:
                    if RandomClip:
                        # get random start
                        if len(images) > seq_len:

                            seg_start = random.randint(0, len(images) - seq_len)
                            images_out = np.array(images[seg_start:seg_start + seq_len])
                        else:
                            images_out = self.fix_ma_Imagesequence(images=images)

                    else:
                        images_out = self.fix_ma_Imagesequence(images=images)
                elif len(images) == 0:

                    all_images = sorted(glob.glob((path + '/' + str(filename) + '**/*.jpg')))

                    # sample the segment
                    frame_start = int(sample[3][:-4])
                    frame_end = int(sample[4][:-4])

                    if frame_start == frame_end:
                        frame_end = frame_end + 2
                    images = all_images[frame_start - 1:frame_end - 1]

                    if len(images) != 0:
                        if RandomClip:
                            # get random start
                            if len(images) > seq_len:

                                seg_start = random.randint(0, len(images) - seq_len)
                                images_out = np.array(images[seg_start:seg_start + seq_len])
                            else:
                                images_out = self.fix_ma_Imagesequence(images=images)

                        else:
                            images_out = self.fix_ma_Imagesequence(images=images)
                    else:
                        print(sample)
                        raise ValueError('Did my best but did not succeed')

            elif data_type in ['features']:

                ###### Call import_pre_extracted ######
                if self.TrainValTest in ['train', 'retrain']:
                    mode_ext = 'Train_Val'
                else:
                    mode_ext = 'Test'

                images_out = import_pre_extracted(sample, images, feat_mod, mode_ext)


                if len(images_out)!=self.seq_length:
                    images_out = self.fix_ma_sequence(temp_arr=images_out)



            else:
                raise ValueError('Unsupported data type, please type "images" or "features"')




        return images_out, sgm_back

##########################################################
#             FIX SEQUENCES
###########################################################
    def fix_ma_Imagesequence(self, images):
        """Given a sequence, fix the size of it to be the one requested.
            Case less: then loop over
            Case more: window sampling
            Case equal: we are good"""
        num_frames = self.seq_length

        count = 0
        # Case where we have less images
        if len(images) < num_frames:
            while len(images) < num_frames:
                count += 1
                # How many more
                extra = num_frames - len(images)
                # If even looping once is not enough
                if extra > len(images):
                    diairesh = extra // len(images)
                    upoloipo = extra % len(images)

                    images_init = images
                    for i in range(diairesh - 1):
                        images = images + images_init

                    if upoloipo > 0:
                        images = images + images[0:upoloipo]

                else:
                    images = images + images[0:extra]
                if count == 300:
                    raise ValueError('You sir are looping eternally!')

            if len(images) > num_frames:
                images1 = images[0:num_frames]
                del images
                images = images1
        elif len(images) > num_frames:
            # Loop until i can be divided with seq length
            looped = (int(np.ceil(len(images) / num_frames)))
            images = images + images[0:looped]

            times = len(images) // num_frames
            mylist = list(images)

            # Temporaly we just avg the features
            new_Ar = [(mylist[i]) for i in range(0, len(mylist), times)]
            images = new_Ar[0:(num_frames)]

        if len(images) != num_frames:
            print(len(images))
            raise ValueError('Issue of expected length persists!')

        return images

    def fix_ma_sequence(self, temp_arr):
        """Given a sequence, fix the size of it to be the one requested.
            Case less: then loop over
            Case more: window sampling
            Case equal: we are good"""
        # If we have fewer frames, then we will loop
        if temp_arr.shape[0] < self.seq_length:
            # how many extra
            extra = self.seq_length - temp_arr.shape[0]
            if extra > temp_arr.shape[0]:
                

                diairesh = extra // temp_arr.shape[0]
                upoloipo = extra % temp_arr.shape[0]

                repeated_div = np.tile(temp_arr, (diairesh, 1))

                loop_im = np.concatenate((repeated_div, temp_arr[0:upoloipo, :]), axis=0)
            else:
                loop_im = temp_arr[0:extra, :]

            images1 = np.concatenate((temp_arr, loop_im), axis=0)

            if len(images1) != self.seq_length:
                raise ValueError('Not enough')

            del temp_arr

            temp_arr = images1
        # If we have more then we need to do window sampling
        elif temp_arr.shape[0] > self.seq_length:

            init_width = temp_arr.shape[1]
            Ap = np.pad(temp_arr, (int(np.ceil(len(temp_arr) / self.seq_length)) * self.seq_length - len(temp_arr), 0),
                        'constant', constant_values=0)
            times = Ap.shape[0] // self.seq_length
            mylist = list(Ap)
            # Temporaly we just avg the features
            new_Ar = np.array([sum(mylist[i:i + times]) / times for i in range(0, len(mylist), times)])
            del temp_arr
            temp_arr = new_Ar[:, 0:init_width]

        assert len(temp_arr) == self.seq_length
        return temp_arr
#########################################################################################
    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""

        out = [process_image(x, self.image_shape) for x in frames]

        return out

    ######################################################################################
    #                                             GENERATORS
    ######################################################################################

    # --------- FOR VIDEO ONLY ----------------------------------
    @threadsafe_generator
    def frame_generator(self, batch_size, data_type):

        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'images'

        """
        # Get the right dataset for the generator
        data = self.data

        print("Creating %s generator with %d samples." % (self.TrainValTest, len(data)))

        # Get the words
        xs, ys, idx_words, _, _ = self.get_words()
        SeenIt = []
        fIN = True

        while data:
            X, y, words, appear, X_words = [], [], [], [], []

            # Reset to be safe.
            sequence, sequence1 = None, None

            # Generate batch_size samples.
            for _ in range(batch_size):

                # Get a random sample.
                if self.TrainValTest in ['train']:

                    sample = random.choice(data)
                    idx = data.index(sample)
                else:
                    if fIN:
                        sample = random.choice(data)
                        SeenIt.append(sample)

                        idx = data.index(sample)
                        data.remove(sample)
                        fIN = False
                    else:
                        sample = random.choice(data)
                        indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        while indices != []:
                            sample = random.choice(data)
                            indices = [i for i, x in enumerate(SeenIt) if x == sample]
                        idx = data.index(sample)
                        SeenIt.append(sample)
                        data.remove(sample)

                sample_words = xs[idx]

                # Check to see if we've already saved this sequence.
                if data_type in ["images"]:
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample, RandomClip=False)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")
                else:
                    raise ValueError('Oops')

                X.append(sequence)
                X_words.append(sample_words)

                y.append(ys[idx])

            appear = np.array(X)
            print(X_words)

            words = np.array(X_words)
            expected = np.array(y)

            yield [words, appear], expected

            del appear, words, expected

            gc.collect
            time.sleep(1)

    @threadsafe_generator
    def frame_generator_Prediction(self, batch_size, data_type):

        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'images'

        """
        # Get the right dataset for the generator
        data = self.data

        print("Creating %s generator with %d samples." % (self.TrainValTest, len(data)))

        # Get the words
        xs, ys, idx_words, _, _ = self.get_words()
        SeenIt = []
        fIN = True

        while data:
            X, y, words, appear, X_words = [], [], [], [], []

            # Reset to be safe.
            sequence, sequence1 = None, None

            # Generate batch_size samples.
            for _ in range(int(batch_size / 2)):

                # Get a random sample.
                if self.TrainValTest in ['train']:
                    # Get the sample id
                    idx = random.randrange(2, len(xs) - 2)

                    sample = data[int(idx / 2) - 1]


                else:
                    if fIN:
                        idx = random.randrange(2, len(xs) - 2)
                        sample = data[int(idx / 2) - 1]
                        SeenIt.append(sample)

                        data.remove(sample)
                        fIN = False
                    else:
                        idx = random.randrange(2, len(xs) - 2)
                        sample = data[int(idx / 2) - 1]

                        indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        while indices != []:
                            idx = random.randrange(2, len(xs) - 2)
                            sample = data[int(idx / 2) - 1]
                            indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        SeenIt.append(sample)
                        data.remove(sample)

                sample_words_verb = xs[int(idx / 2)]
                sample_words_noun = xs[int(idx / 2) + 1]

                # Check to see if we've already saved this sequence.
                if data_type in ["images"]:
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)


                elif data_type in ['features']:
                    sequence = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type, feat_mod='obj')



                else:
                    raise ValueError('Oops')

                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                # Verb append
                X.append(sequence)

                X_words.append(sample_words_verb)

                if ys[idx] == 0 or ys[idx + 1] == 0:
                    raise ValueError('whate')
                y.append(ys[idx])

                # Object append
                X.append(sequence)
                X_words.append(sample_words_noun)
                y.append(ys[idx + 1])

            appear = np.array(X)

            words = np.array(X_words)
            expected = np.array(y)

            yield [words, appear], expected

            del appear, words, expected

            gc.collect
            time.sleep(1)

    ########################################################################################################################
    # This generator produces two tasks for each sample (i.e. produces two samples for the same thing)
    # 1. the goal is to predict the verb class given a) frames of the previous segment, b) words up until the previous sgement
    # 2. the goal is now to predict the object, given a) frames of the previous segment, b) words up until the verb of the current
    @threadsafe_generator
    def frame_generator_PredictionNLP_original(self, batch_size, data_type, what2Do):

        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'images'

        """
        # Get the right dataset for the generator
        data = self.data

        print("Creating %s generator with %d samples." % (self.TrainValTest, len(data)))

        # Get the words
        xs, ys, idx_words, _, _ = self.get_words()
        SeenIt = []
        fIN = True
        y_true = []

        while len(SeenIt) < len(data):
            X, y, words, appear, X_words = [], [], [], [], []

            # Reset to be safe.
            sequence, sequence1 = None, None

            # Generate batch_size samples.
            for _ in range(int(batch_size / 2)):

                # Get a random sample.
                if self.TrainValTest in ['train']:
                    # Get the sample id
                    idx = random.randrange(2, len(xs) - 2)

                    sample = data[int(idx / 2) - 1]


                else:
                    if fIN:
                        idx = random.randrange(2, len(xs) - 2)
                        sample = data[int(idx / 2) - 1]
                        SeenIt.append(sample)

                        fIN = False
                    else:
                        idx = random.randrange(2, len(xs) - 2)
                        sample = data[int(idx / 2) - 1]

                        indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        while indices != []:
                            idx = random.randrange(2, len(xs) - 2)
                            sample = data[int(idx / 2) - 1]
                            indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        SeenIt.append(sample)

                sample_words_verb = xs[int(idx / 2)]
                sample_words_noun = xs[int(idx / 2) + 1]

                # Check to see if we've already saved this sequence.
                if data_type in ["images"]:
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)


                elif data_type in ['features']:
                    sequence = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type)
                else:
                    raise ValueError('Oops')

                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                # Verb append
                X.append(sequence)
                X_words.append(sample_words_verb)
                y.append(ys[idx])

                # Object append
                X.append(sequence)
                X_words.append(sample_words_noun)
                y.append(ys[idx + 1])

            appear = np.array(X)

            words = np.array(X_words)
            expected = np.array(y)
            y_true.append(expected)
            if what2Do in ['NLP']:
                yield words, expected
            elif what2Do in ['Vid']:
                yield appear, expected
            else:

                yield [words, appear], expected

            del appear, words, expected

            if self.TrainValTest in ['test']:
                np.save('y_true.npy', y_true)
            gc.collect
            time.sleep(1)

        ########################################################################################################################
        # This generator only allows for object prediction, given a) Frames of the previous segment, b) words up until the verb
        # for the current segment
    @threadsafe_generator
    def frame_generator_PredictionNLP_Action(self, batch_size, data_type, what2Do):

            """Return a generator that we can use to train on. There are
            a couple different things we can return:

            data_type: 'images'

            """
            # Get the right dataset for the generator
            data = self.data

            print("Creating %s generator with %d samples." % (self.TrainValTest, len(data)))

            # Get the words
            xs, ys, idx_words, _, _, _, SeqSamp_Ids = self.get_words_ActionPrediction_Action()

            SeenIt, y_true = [], []
            fIN = True

            while len(SeenIt) < len(data):
                X, y, words, appear, X_words = [], [], [], [], []
                X_gaze, X_hands, y_noun, X_TSN = [], [], [], []
                # Reset to be safe.
                sequence, sequence1 = None, None

                # Generate batch_size samples.
                for _ in range(int(batch_size)):

                    # Get a random sample.
                    if self.TrainValTest in ['train']:
                        # Get the sample id
                        idx = random.randrange(0, len(data) - 1)


                        sample = data[idx]


                    else:
                        if fIN:
                            idx = random.randrange(0, len(data) - 1)

                            sample = data[idx]
                            SeenIt.append(sample)

                            fIN = False
                        else:
                            idx = random.randrange(0, len(data) - 1)

                            sample = data[idx]

                            indices = [i for i, x in enumerate(SeenIt) if x == sample]

                            while indices != []:
                                idx = random.randrange(0, len(data) - 1)

                                sample = data[idx]
                                indices = [i for i, x in enumerate(SeenIt) if x == sample]

                            SeenIt.append(sample)

                    # Check to see if we've already saved this sequence.
                    if data_type in ["images"]:
                        # Get and resample frames.
                        frames, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type,
                                                                      timesteps=True)

                        # Build the image sequence
                        sequence = self.build_image_sequence(frames)

                    elif data_type in ['features']:

                        sequence, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type,
                                                                        feat_mod='obj')

                        if self.model_name in ['ObjsGazeHands_Only', 'ObjsGazeHands_NLP']:
                            sequence_gaze, sgm_back = self.get_frames_for_sample(sample, RandomClip=True,
                                                                                 data_type=data_type, feat_mod='gaze')

                            sequence_hands, sgm_back = self.get_frames_for_sample(sample, RandomClip=True,
                                                                                  data_type=data_type, feat_mod='hands')

                        elif self.model_name in ['ObjsGazeTSNRGB_Only', 'ObjsGazeTSNRGB_NLP']:

                            sequence_gaze, sgm_back = self.get_frames_for_sample(sample, RandomClip=True,
                                                                                 data_type=data_type, feat_mod='gaze',
                                                                                 timesteps=True)
                            sequence_TSN, sgm_back = self.get_frames_for_sample(sample, RandomClip=True,
                                                                                data_type=data_type,
                                                                                feat_mod='tsn_flow', timesteps=True)


                    else:
                        raise ValueError('Spmething is Wrong!')

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                    # Verb append
                    X.append(sequence)


                    sample_words_action = xs[idx]

                    if self.model_name in ['ObjsGazeHands_Only', 'ObjsGazeHands_NLP']:
                        X_gaze.append(sequence_gaze)
                        X_hands.append(sequence_hands)
                    if self.model_name in ['ObjsGazeTSNRGB_NLP', 'ObjsGazeTSNRGB_Only']:
                        X_gaze.append(sequence_gaze)
                        X_TSN.append(sequence_TSN)


                    X_words.append(sample_words_action[:-1])
                    y.append(ys[idx])




                appear = np.array(X)
                if self.model_name in ['ObjsGazeHands_Only', 'ObjsGazeHands_NLP']:
                    gaze = np.array(X_gaze)
                    hands = np.array(X_hands)

                if self.model_name in ['ObjsGazeTSNRGB_NLP', 'ObjsGazeTSNRGB_Only']:
                    gaze = np.array(X_gaze)
                    TSN = np.array(X_TSN)

                words = np.array(X_words)
                expected = np.array(y)
                y_true.append(expected)


                if what2Do in ['NLP']:
                    yield words, expected
                elif what2Do in ['Vid', 'Obj_Only']:
                    yield appear, expected
                elif what2Do in ['ObjsGazeHands_Only']:
                    yield [appear, gaze, hands], expected
                elif what2Do in ['ObjsGazeHands_NLP']:
                    yield [words, appear, gaze, hands], expected
                elif what2Do in ['ObjsGazeTSNRGB_NLP']:
                    yield [words, appear, gaze, TSN], expected
                elif what2Do in ['ObjsGazeTSNRGB_Only']:
                    yield [appear, gaze, TSN], expected

                else:

                    yield [words, appear], expected

                del appear, words, expected

                if self.TrainValTest in ['test']:


                    np.save('y_true.npy', y_true)
                    np.save('y_predSamples.npy', np.array(SeenIt))
                gc.collect

                time.sleep(1)


    ########################################################################################################################
    # This generator only allows for object prediction, given a) Frames of the previous segment, b) words up until the verb
    # for the current segment
    @threadsafe_generator
    def frame_generator_PredictionNLP(self, batch_size, data_type, what2Do, PoS):

        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'images'

        """
        # Get the right dataset for the generator
        data = self.data

        print("Creating %s generator with %d samples." % (self.TrainValTest, len(data)))

        # Get the words
        xs, ys, idx_words, _, _, _,SeqSamp_Ids = self.get_words_ActionPrediction()

        SeenIt, y_true,y_true_noun = [], [],[]
        fIN = True


        while len(SeenIt) < len(data):
            X, y, words, appear, X_words = [], [], [], [], []
            X_gaze, X_hands, y_noun, X_TSN = [], [], [], []
            # Reset to be safe.
            sequence, sequence1 = None, None

            # Generate batch_size samples.
            for _ in range(int(batch_size)):

                # Get a random sample.
                if self.TrainValTest in ['train']:
                    # Get the sample id
                    idx = random.randrange(0, len(data) - 1)

                    sample = data[idx]


                else:
                    if fIN:
                        idx = random.randrange(0, len(data) - 1)

                        sample = data[idx]
                        SeenIt.append(sample)

                        fIN = False
                    else:
                        idx = random.randrange(0, len(data) - 1)

                        sample = data[idx]

                        indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        while indices != []:
                            idx = random.randrange(0, len(data) - 1)

                            sample = data[idx]
                            indices = [i for i, x in enumerate(SeenIt) if x == sample]

                        SeenIt.append(sample)



                # Check to see if we've already saved this sequence.
                if data_type in ["images"]:
                    # Get and resample frames.
                    frames, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type, timesteps = True)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)

                elif data_type in ['features']:

                    sequence, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type, feat_mod='obj')

                    if self.model_name in ['ObjsGazeHands_Only', 'ObjsGazeHands_NLP']:
                        sequence_gaze, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type,feat_mod='gaze')

                        sequence_hands, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type,feat_mod='hands')

                    elif self.model_name  in ['ObjsGazeTSNRGB_Only', 'ObjsGazeTSNRGB_NLP']:

                        sequence_gaze, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type, feat_mod='gaze',timesteps = True)
                        sequence_TSN, sgm_back = self.get_frames_for_sample(sample, RandomClip=True, data_type=data_type,  feat_mod='tsn_flow',timesteps = True)


                else:
                    raise ValueError('Oops')

                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")

                # Verb append
                X.append(sequence)
                print('----------')
                print(idx)
                print(sgm_back)
                print(len(xs))
                print(len(SeqSamp_Ids))
                print('++++++++++++')
                sample_words_verb = xs[SeqSamp_Ids[idx-sgm_back][0]]

                sample_words_noun = xs[SeqSamp_Ids[idx-sgm_back][1]]

                if self.model_name in ['ObjsGazeHands_Only', 'ObjsGazeHands_NLP']:
                    X_gaze.append(sequence_gaze)
                    X_hands.append(sequence_hands)
                if self.model_name in ['ObjsGazeTSNRGB_NLP', 'ObjsGazeTSNRGB_Only']:
                    X_gaze.append(sequence_gaze)
                    X_TSN.append(sequence_TSN)

                if PoS == 'verb' and self.model_name not in ['VNLP_Both']:
                    X_words.append(sample_words_verb)
                    y.append(ys[SeqSamp_Ids[idx][0]])

                if PoS == 'noun' and self.model_name not in ['VNLP_Both']:
                    X_words.append(sample_words_noun)
                    y.append(ys[SeqSamp_Ids[idx][1]])

                if  self.model_name in ['VNLP_Both']:

                    X_words.append(sample_words_noun)
                    y.append(ys[SeqSamp_Ids[idx][0]])
                    y_noun.append(ys[SeqSamp_Ids[idx][1]])

            appear = np.array(X)
            if self.model_name in ['ObjsGazeHands_Only', 'ObjsGazeHands_NLP']:
                gaze = np.array(X_gaze)
                hands = np.array(X_hands)

            if self.model_name in ['ObjsGazeTSNRGB_NLP', 'ObjsGazeTSNRGB_Only']:
                gaze = np.array(X_gaze)
                TSN = np.array(X_TSN)

            words = np.array(X_words)
            expected = np.array(y)
            y_true.append(expected)
            if self.model_name in ['VNLP_Both']:
                expected_noun = np.array(y_noun)
                y_true_noun.append(expected_noun)

            if what2Do in ['NLP']:
                yield words, expected
            elif what2Do in ['Vid', 'Obj_Only']:
                yield appear, expected
            elif what2Do in ['ObjsGazeHands_Only']:
                yield [appear, gaze, hands], expected
            elif what2Do in ['ObjsGazeHands_NLP']:
                yield [words, appear, gaze, hands], expected
            elif what2Do in ['ObjsGazeTSNRGB_NLP']:
                yield [words, appear, gaze, TSN], expected
            elif what2Do in ['ObjsGazeTSNRGB_Only']:
                yield [appear, gaze, TSN], expected
            elif what2Do in ['VNLP_Both']:

                yield [words, appear], [expected, expected_noun]
            else:

                yield [words, appear], expected

            del appear, words, expected

            if self.TrainValTest in ['test']:

                if what2Do in ['VNLP_Both']:
                    np.save('y_true_verb.npy', y_true)
                    np.save('y_true_noun.npy', y_true_noun)
                else:
                    np.save('y_true.npy', y_true)
                np.save('y_predSamples.npy', np.array(SeenIt))
            gc.collect

            time.sleep(1)

    ########################################################################################################################
    # It does the same job as frame_generator_Prediction_original but supports multi-gpu designs

    def frame_generator_PredictionNLPMULTI(self, batch_size, data_type, what2Do):

        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'images'

        """
        # Get the right dataset for the generator
        data = self.data

        print("Creating %s generator with %d samples." % (self.TrainValTest, len(data)))

        # Get the words
        xs, ys, idx_words, _, _ = self.get_words()
        SeenIt = []
        fIN = True
        X, y, words, appear, X_words = [], [], [], [], []
        count = 0
        while count < 21:
            count += 1
            # Reset to be safe.
            sequence, sequence1 = None, None

            # Generate batch_size samples.

            # Get a random sample.
            if self.TrainValTest in ['train']:
                # Get the sample id
                idx = random.randrange(2, len(xs) - 2)

                sample = data[int(idx / 2) - 1]


            else:
                if fIN:
                    idx = random.randrange(2, len(xs) - 2)
                    sample = data[int(idx / 2) - 1]
                    SeenIt.append(sample)

                    data.remove(sample)
                    fIN = False
                else:
                    idx = random.randrange(2, len(xs) - 2)
                    sample = data[int(idx / 2) - 1]

                    indices = [i for i, x in enumerate(SeenIt) if x == sample]

                    while indices != []:
                        idx = random.randrange(2, len(xs) - 2)
                        sample = data[int(idx / 2) - 1]
                        indices = [i for i, x in enumerate(SeenIt) if x == sample]

                    SeenIt.append(sample)
                    data.remove(sample)

            sample_words_verb = xs[int(idx / 2)]
            sample_words_noun = xs[int(idx / 2) + 1]

            # Check to see if we've already saved this sequence.
            if data_type in ["images"]:
                # Get and resample frames.
                frames = self.get_frames_for_sample(sample, RandomClip=False)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")
            else:
                raise ValueError('Oops')

            # Verb append
            X.append(sequence)
            X_words.append(sample_words_verb)
            y.append(ys[idx])

            # Object append
            X.append(sequence)
            X_words.append(sample_words_noun)
            y.append(ys[idx + 1])

        appear = np.array(X)

        words = np.array(X_words)
        expected = np.array(y)

        if what2Do in ['NLP']:
            return words, expected
        elif what2Do in ['Vid']:
            return appear, expected
        else:

            return [words, appear], expected

        del appear, words, expected

        gc.collect
        time.sleep(1)
