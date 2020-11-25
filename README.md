# Audio-Recognition-with-CNN

data set : http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip

### Data Preprocessing
#### 1) split data into train, test and val
rename the unzip folder (/mini_speech_commands) as /data/train. after executing **split_train_val_test** function it will create 2 more directories 
(/data/val and /data/test) which contain train, test and validation splitted data from all classes.

#### 2) encode audio file to tensor 
using **tf.io.read_file(filename)** each audio file needs to be convert into binary audio data file. then the binary file can be decode into a tensor using 
**tf.audio.decode_wav(audio_binary)**

#### 3) Convert 1D signal into 2D
the decoded tensor is a vector. So this signal vector can be convered to signal matrix using fourier transform. Using Fourier transform it can convert time 
series signal into frequency spectrum and which is 2 dimensional signal.
