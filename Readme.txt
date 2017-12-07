This directory contains everything I downloaded and used to make the 
gender-discriminating model.

REQUIREMENTS:
This project required tensorflow, and numpy

DOWNLOADED DATA:
The image dataset
./Data/aligned/ as well as ./Data/Folds/original_txt_files
link: http://www.openu.ac.il/home/hassner/Adience/data.html#agegender

I deleted ./Data/aligned/ in order to make the upload less than 2GB 
as required by dropbox. You will need to copy the data files from 
aligned.tar.gz to that location.

The VGG_Face model
./Models/VGG_Face_Caffe/
link: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz

DOWNLOADED CODE:
I downloaded caffe-tensorflow to help with the model conversion
link: https://github.com/ethereon/caffe-tensorflow

I only used the .npy file in generating my model, but referred to the
author's code for help debugging issues with my TF code. 

I also used code from the lab that created the age and gender dataset
to format the training, validation, and test files and gender labels.
./Data/DataPreparationCode
link: https://github.com/GilLevi/AgeGenderDeepLearning

I also made use of some code snippets from tensorflow.org

STEPS TAKEN:

The converter uses a newer version of caffe than the original
prototxt, so the prototxt had to be mofified (mostly changing 
capitalization). 
Documented here: https://github.com/ethereon/caffe-tensorflow/issues/39

All of the paths in ./Data/DataPreparationCode/create_train_val_txt_files.py 
had to be modified, and the pointers to files updated to use the
frontally aligned faces.

preparation.sh contains commands to convert the VGG_Face model 
weights into a numpy file for easier reading. The weights are located
at ./Models/VGG_face_TF

preparation.sh also has commands to create the new training, 
validation, and test image sets.

I then wrote VGG_FACE_conversion_test.py to ensure that I had copied
the VGG_Face model correctly. Alternating between the caffe model and
the tensorflow model was useful in debugging here. Because I was 
slightly modifying the way I used the model in other files I opted 
not to write the model as a class yet. 

Next I precomputed features from the VGG_face model. Because of the 
computer I am working on now is rather old and lacks a GPU, I opted
to precompute all the way up through fc7, meaning that I only needed
to finetune a single layer. Running the following command

 ./precompute_features.py 

will precompute the VGG_face features for all the images, using fold0
as the test set.

The file train_gender_model.py generates the final layer of the model
and trains it using the precomputed training and validation sets. I
used a form of early stopping to get decent validation accuracy.

I computed the test accuracy using test_gender_layer.py. Here I was
able to make use of the precomputed test features. The accuracy value
I got was 90.46%, which is not that great. On cases where the input
image was male the accuracy was 91.92%, and on female inputs the 
accuracy was 88.93%.

Given more time/resources I would look into using the features from 
the pool5 layer, as the features in the final layer are most likely 
already directed towards classifying identity rather than gender. 
Going into this project I expected this not to be the case.

Run the following commands to test the final model on a 816x816 
frontally aligned RGB images. 

python ./final_gender_model.py  ./femaletest.jpg
python ./final_gender_model.py  ./maletest.jpg
python ./final_gender_model.py  ./testimg.jpg

Thanks for your time and let me know if there are any issues at 
danny.jeck@gmail.com



