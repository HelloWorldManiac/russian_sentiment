# Russian short messages sentiment classifier

This model is derived to get rid of dictionary-based approach, which is crucial for Russian language known by its rich affixation.
## Input data
- short texts
## Target
- labels denoting positive, negative or neutral tonality of a message

## Data sources:
- labelled data from [vk.com](https://aclanthology.org/C18-1064.pdf)
- labelled data from Russian segment of [Twitter](https://twitter.com/)
- labelled data from [otzovik](https://otzovik.com/)

Due to machine labelling a substantial part of messages has been mislabelled. In order to compose a reliable dataset the initial data has been distilled to ~33k items.
Another problem was the strong correlation between labels and the presence of emoticons in the text. In order to overcome this semiotic shift (thus, overfitting thereto) and to base entirely upon the semantics of the texts the most popular emoticons have been filtered out.

## Preprocessing
The preprocessing process is reduced to two stages:

- elimintaion of emoticons, ids and notorious punctuation marks
- tokenization and encoding with old but gold Tensorflow solution [SubwordTextEncoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder)
By using this approach we reduce the vocab to 2^12 items without losing the ability to catch the semantics of a certain sequence. Therefore we do not need any stemming/lemmatization any more as well we have no need to coin a specific policy for OOVs since any word is bound to be split into a sequence of tokens.

## Model

A number of architectures has been tested (MLP, CNN, LSTM, BiLSTM). The best result has been achieved with the BiLSTM.

## Metrics
The dataset has been split into train/test in proportion 85/15. Then the model has been fit on the training subset with the validation split of 15%.

| Metric | Value |
| ------ | ------ |
| Validation Accuracy | 0.8119 |
| F1-score (macro) on test | 0.8015 |
| avg CPU time| < 0.015s |

In order to reproduce this result on an adequate dataset the weights of the model have been saved and then used in the Flask application.

## Application

A dockerised application has been developed as an MVP.
In order to build it run:

```sh
docker build -t [name] .
```
In order to run it in a detached mode run
```sh
docker run  -d -p 8585:8585 [name]
```
To verify its work please run
```sh
curl -X POST -d 'text=синхрофазотрон' http://localhost:8585/sentiment
```
## License

MIT



