
# coding: utf-8

# In[1]:


import glob
from random import shuffle

import librosa
import numpy as np
import pandas as pd
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# In[2]:


input_length = 16000*5

batch_size = 32


n_mels = 320
# def audio_norm(data):
#
#     max_data = np.max(data)
#     min_data = np.min(data)
#     data = (data-min_data)/(max_data-min_data+0.0001)
#     return data-0.5

def preprocess_audio_mel_T(audio, sample_rate=16000, window_size=20, #log_specgram
                 step_size=10, eps=1e-10):

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels= n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40

    return mel_db.T


def stretch(data, rate=1):
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def pitch_shift(data, n_steps=3.0):
    data = librosa.effects.pitch_shift(data, sr=input_length, n_steps=n_steps)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def loguniform(low=0.00000001, high=0.01):
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def white(N, state=None):
    """
    White noise.

    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`

    White noise has a constant power density. It's narrowband spectrum is therefore flat.
    The power in white noise will increase by a factor of two for each octave band,
    and therefore increases with 3 dB per octave.
    """
    state = np.random.RandomState() if state is None else state
    return state.randn(N)


def augment(data):
    if np.random.uniform(0, 1) > 0.90:
        wnoise = loguniform()
        data = data + wnoise * white(len(data))
    if np.random.uniform(0, 1) > 0.90:
        stretch_val = np.random.uniform(0.9, 1.1)
        data = stretch(data, stretch_val)
    if np.random.uniform(0, 1) > 0.90:
        pitch_shift_val = np.random.uniform(-6, 6)
        data = pitch_shift(data, n_steps=pitch_shift_val)
    return data


def load_audio_file(file_path, input_length=input_length, aug=False):
    data = librosa.core.load(file_path, sr=16000)[0]  # , sr=16000
    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset:(input_length + offset)]


    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    if aug:
        data = augment(data)

    data = preprocess_audio_mel_T(data)

    return data


# In[3]:


train_files = glob.glob("../input/audio_train/*.wav")
test_files = glob.glob("../input/audio_test/*.wav")
train_labels = pd.read_csv("../input/train.csv")


# In[4]:


file_to_label = {"../input/audio_train/"+k:v for k,v in zip(train_labels.fname.values, train_labels.label.values)}


# In[5]:


#file_to_label


# In[7]:

#
# data_base = load_audio_file(train_files[0])
# fig = plt.figure(figsize=(14, 8))
# plt.title('Raw wave : %s ' % (file_to_label[train_files[0]]))
# plt.ylabel('Amplitude')
# plt.plot(np.linspace(0, 1, input_length), data_base)
# plt.show()


# In[8]:


list_labels = sorted(list(set(train_labels.label.values)))


# In[9]:


label_to_int = {k:v for v,k in enumerate(list_labels)}


# In[10]:


int_to_label = {v:k for k,v in label_to_int.items()}


# In[11]:


file_to_int = {k:label_to_int[v] for k,v in file_to_label.items()}


# In[12]:


def get_model_mel():

    nclass = len(list_labels)

    inp = Input(shape=(157, 320, 1))

    incep_res = InceptionResNetV2(input_shape=(157, 320, 1), weights=None, include_top=False)

    x = incep_res(inp)

    x = GlobalMaxPool2D()(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(128, activation=activations.relu)(x)
    x = Dense(nclass, activation=activations.softmax)(x)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model


# In[13]:


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# In[14]:


def train_generator(list_files, batch_size=batch_size, aug=False):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath, aug=aug) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, :,np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)
            
            yield batch_data, batch_labels
            


# In[15]:


tr_files, val_files = train_test_split(sorted(train_files), test_size=0.1, random_state=42)


# In[16]:


model = get_model_mel()
model.load_weights("incres.h5")
#
# # In[17]:
#
#
model.fit_generator(train_generator(tr_files, aug=True), steps_per_epoch=len(tr_files)//batch_size, epochs=100,
                    validation_data=train_generator(val_files), validation_steps=len(val_files)//batch_size,
                   use_multiprocessing=True, workers=8, max_queue_size=20,
                    callbacks=[ModelCheckpoint("incres.h5", monitor="val_acc", save_best_only=True),
                               EarlyStopping(patience=5, monitor="val_acc")])




#model.save_weights("baseline_cnn.h5")
model.load_weights("incres.h5")

# In[19]:





# In[20]:
bag = 10

array_preds = 0

for i in tqdm(range(bag)):

    list_preds = []

    for batch_files in tqdm(chunker(test_files, size=batch_size), total=len(test_files)//batch_size ):
        batch_data = [load_audio_file(fpath) for fpath in batch_files]
        batch_data = np.array(batch_data)[:, :, :,np.newaxis]
        preds = model.predict(batch_data).tolist()
        list_preds += preds



    # In[21]:


    array_preds += np.array(list_preds)/bag


# In[22]:


list_labels = np.array(list_labels)


# In[30]:


top_3 = list_labels[np.argsort(-array_preds, axis=1)[:, :3]] #https://www.kaggle.com/inversion/freesound-starter-kernel
pred_labels = [' '.join(list(x)) for x in top_3]


# In[31]:


df = pd.DataFrame(test_files, columns=["fname"])
df['label'] = pred_labels


# In[32]:


df['fname'] = df.fname.apply(lambda x: x.split("/")[-1])


# In[33]:


df.to_csv("inceptionresnet.csv", index=False)

