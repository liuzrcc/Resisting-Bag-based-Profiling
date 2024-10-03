This repository contains the code of our paper "Resisting Bag-based Attribute Profiling by Adding Items to Existing Profiles".

## Data Preparation.

Emotion Profile data set is curated by combining two data sets, namely [Emotion6](http://chenlab.ece.cornell.edu/downloads.html)[1] and [Emotion benchmark paper](https://arxiv.org/pdf/1605.02677)[2].
Please refer to list form [data structure](https://github.com/liuzrcc/Resisting-Bag-based-Profiling/blob/main/data/Emotion_Profile_Data).

For Personality Profile, our data set is based on [PsychoFlickr](http://www.cristinasegalin.com/research/projects/phd/personality/PsychoFlickr_images.rar)[3]. The exact data split is introduced in the paper. 


## Train deep bag-based MIL models.

You can select a MIL architecture from ```sett, deepset, attmil, or cnnvote```.

To train a bag-based MIL classifier,

```
python3 tainMIL.py --method sett --Emocat anger
```

## Attack bag-based classifiers

To attack a trained bag-based classifier,

```
python3 blackbox_attack.py --method sett --Emocat anger
```




[1] A mixed bag of emotions: Model, predict, and transfer emotion distributions. CVPR 2015.

[2] Building a large scale dataset for image emotion recognition: The fine print and the benchmark. AAAI 2016.

[3] Social profiling through image understanding: Personality inference using convolutional neural networks. CVIU 2016.


