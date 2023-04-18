# Supervised Learning Analysis

The purpose of this project is to explore techniques in supervised learning. It is important to realize that understanding an algorithm or technique requires understanding how it behaves under a variety of circumstances. As such, I will implement the several learning algorithms and compare their performance.

I implemented five learning algorithms. They are for:
- **Decision trees with some form of pruning**
- **Neural networks**
- **Boosting**
- **Support Vector Machines**
- **knearest**
- **neighbors**

I selected two classification problems (Speech Emotion Recognition and Forest Cover Type) and analyze their behaviors in different learning algorithms.


## Speech Emotion Recognition

Speech Emotion Recognition is a task of recognizing human’s emotion from speech regardless of the speech contents. We have being always want to achieve the natural communication between human and machine just like the communication between humans. After years of research, we are able to converting the human speech into a sequence of words. However, despite the great progress made in speech recognition, we are still far from having a natural interaction between man and machine because the machine does not understand the emotional state of the speaker[1]. That’s why I want to pick the problem of speech emotion recognition(SER). It’s important to let the machine have enough intelligence to understand human emotion by voice.

**Data Processing**
The rawest datasets I started with is a collection of 3 seconds audio files in which different actors repeat similar sentences with different emotions. Then I processed the audios with python audio analyzing library Librosa, and extracted three features:

- MFCC (Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound).
- Chroma ( Pertains to the 12 different pitch classes).
- Mel (Mel Spectrogram Frequency).

Each feature provides some attributes for further classification. The challenge for this classification problem is, the datasets are not very large (around 1300), while there are 180 attributes and 8 classes. To simplify the task I reduced the classes to 4 emotions that might be easier to identify: calm, happy, fearful, and disgust. The training size shrinks to 768. It would be a challenge to implement the learning algorithms to achieve good learning results.

Audio Data:
https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view

## Forest Cover Type

Predicting forest cover type is learning the forest cover type classes and it’s cartographic variables, and predict the cover type in the future. The forest in this study are wildness area with minimal human-caused disturbances. To do inventory of natural resources in a wild area can be very difficult but also really important. It helps to observe ecological changes and identify the type of biosphere within the area. Generally, cover type data is either directly recorded by field personnel or estimated from remotely sensed data. Both of these techniques may be prohibitively time consuming and/or costly in some situations[2]. Thus we can use the predictive forest cover type models to obtain such data efficiently.

**Data Processing**
This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form and contains binary columns of data for qualitative independent variables (wilderness areas and soil types)[3].

The original datasets have more than 580,000 instances, and 54 attributes. To build the model efficiently, in the following learning algorithm studies, I randomly selected around 3000 instances to train the models.

Data Set:  [https://archive.ics.uci.edu/ml/datasets/Covertype]
