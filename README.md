# Dank Meme Rating
This repository contains the codes for the **Neural Network** model for upvote/like prediction of memes using **transfer learning**. This was done as a part of Neural Networks and Applications [course](https://eee.nitk.ac.in/course/neural-networks-and-applications-ee319) at NITK.\
Link to the project [report](https://meghbhalerao.github.io/pdfs/Megh-Bhalerao-Dank-Meme-Rating-Report.pdf). 

## Brief Description of the project
Here, we train a neural network model (Google [InceptionV4 Net](https://research.google/pubs/pub45169/)) to classify a **meme image** whether it is _dank_ or not. A meme is said to _dank_ if the rating that the model predicts is greater than 0.5.
## Steps to run the code
1. Download the dataset from the link given in `./dataset_link.txt`
2. To extract the features from the meme images, run `python ./utils/feature_extraction.py`
3. To train the model on the extracted features, run `python train.py`
4. To do inference on an unseen dataset, run `python eval.py`
## Dependencies
- [`tensorflow`](https://www.tensorflow.org)
- [`matplotlib`](https://matplotlib.org)


