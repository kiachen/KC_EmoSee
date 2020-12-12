Project Title: Face Expression Recognition
Used Dataset: FER2013

EmoSee is a program that is able to recognize human face expression and analyst the emotion shown from the images capture by computer vision. EmoSee was using Dlib(python library) to identify the frontal faces via some facial landmarks and classify the extracted image information into seven types of emotions which are happy, neutral, surprised, sad, angry, disgust, scared. The EmoSee program contain a emotion model classifier which can analyst and predict the emotion from human face with the pretrained model on FER2013 dataset. The data in FER 2013 dataset consists a huge amount of labeled human's emotions samples.

Libraries needed(require to download):
Pillow==8.0.1
opencv==4.4.0
numpy==1.18.5
dlib==19.21.0(visual studio c++ package, cmake)
Keras==2.4.3
tensorflow==2.3.1
