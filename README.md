# Data Science

## Introduction

**This README section is the shortened version of `main.ipynb` notebook. It's still contains all the main information, but for more full and detailed comments on topics, code and etc, please, refer to notebook file in `notebooks/` folder.**

This project works with dataset for binary sentiment classification. It provides a set of 50,000 polar movie reviews for training and testing.

Firstly, download and import everything that will be used

## EDA

Check for imbalance:

![alt text](assets/image.png)

Dataset is perfectly balanced

## Feature engineering

### Data cleaning

As a first step, i will remove any redundant characters, that doesnt provide any value for sentiment analysis: special characters, single characters, etc:

```
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters from start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with single space
    text = re.sub(r'^b\s+', '', text)  # Remove prefixed 'b'
    text = text.lower()
    return text
```

### Tokenization

Now, I will perform tokenization. While there is much different methodics to tokenize, I decided that simple tokenization word by word is good enough for the task:

### Stop-words filtering

Now, removing stopwords. Applies the same logic as in removing characters: removing that ones, which will not provide any value for sentiment analysis, but now we will use predefined downloaded set of stopwords.

### Stemming vs Lemmatization

Stemming and lemmatization are both techniques used to reduce words to their base or root form. Stemming is a rule-based process that removes suffixes to get the root form of a word, often leading to non-real words.Lemmatization reduces words to their dictionary form (lemma) using linguistic knowledge, ensuring real words.

To decide which one to use, I tried both.

After trying, i came to this conclusion. While lemmatization has left some wordds unchanged (like caught, learnt), it provided much more accurate results, while stemming often made not existing words (which is really important considering I will later try word embedding), and at the same time, lemmatization was ~3 times faster for me. That leaves no room for any reason to not prefer lemmatization over stemming

### Vectorization

Machine learning models cant work with strings: they firstly need to be converted to some numerical form. I will try to different methods of vectorization: Bag-of-Words and Word Embedding. Bag-of-Words is a simple, frequency-based text representation where each word is treated as an independent feature. Word embedding is on the other hand, a dense vector representation of words that captures semantic meaning and relationships, that positively distinguishes this method from the previous one. While its pros are obvious, the cons are that it requires more computation, and needs either big dataset to train effectively or pretrained models. In my case, i used one called Word2Vec.

Conclusions: for sentiment analysis, Word embeding is much, much more suitable choice, so I will go on with it

## Modeling

I tried 4 different models:

### Logistic regression

Accuracy: 0.8554
My local runtime: ~ 1 second

### Random Forest Classifier

Accuracy: 0.8100

My local runtime: ~ 70 seconds

### SVM

Accuracy: 0.8629

My local runtime: ~ 270 seconds

### Deep learning

Accuracy: 0.8665

My local runtime: ~ 75 seconds

### Conclusions

For my final project (MLE part) I will choose Logistic Regressions. It's simple, has a accuracy score not much less than DL model or SVM (0.8554 against ~0.86), and extremely fast.

Of course, my results aren't 100% accurate. I could have played more with hyperparameters tuning, try more DL model architectures (I actually did it but left it all behind in final notebook version). I could have tried more vectorization methods, like TF-IDF. But at the end, chosen model completes our task — sentiment analysis with accuracy > 0.85, so I will b satisfied for now with this result.



# MLE

## Prerequisites

### Forking and Cloning from GitHub

To start using this project, you first need to get a local copy of this project. U can fork this project to your account, using "Fork" button at the top right part of the projects page, and then get the local copy of this project using `git clone`. After that, you should have this project on your machine with your work directory being project root.

### Docker

This project can be run in two ways: by running all related scripts using IDE or other instrument that allows to run python scripts, or by building Docker images and running Docker containers later on. To run using Docker, you need to install ([Docker Desktop](https://www.docker.com/products/docker-desktop)) — it is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. 

## Running project

Each part of project is logging it's actions in the terminal. You can check it to see if script finished or something went wrong.**Note, that model metrics will also appear in logs**

### Data loading

Firstly, you need to get data to train your model. Simply go to `src/` folder and run `data_loader.py` script — after finishing, new folder named `data` will be created in `src` folder, and all necessary data will be downloaded from cloud.

### Train

To train the model using Docker: 

- Build the training Docker image:
```bash
docker build -f ./src/train/Dockerfile -t sentiment-analysis .  
```
- Then, run the container to train your model. After finishing, container will automatically remove itself and you should have model on your local machine in `outputs/models` folder:
```bash
docker run --rm -v ${PWD}/outputs/models:/app/outputs/models sentiment-analysis
```

Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 src/train/train.py
```

### Inference

Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

- Build the inference Docker image:
```bash
docker build -f ./src/inference/Dockerfile -t sentiment-inference . 
```

- Then, run the container to inference on new, unseen data . After finishing, container will automatically remove itself and you should have inference results on your local machine in `outputs/results` folder:
```bash
docker run --rm -v ${PWD}/outputs:/app/outputs sentiment-inference
```

Alternatively, the `inference.py` script can also be run locally as follows:

```bash
python3 src/train/inference.py
```

P.S: Considering that to test our inference we were guided to use the same test dataset we used, well, for test, the `inference.py` script detects if there is a sentiment column in inference dataset. If there is, it will also log accuracy based on predicted vs true labels