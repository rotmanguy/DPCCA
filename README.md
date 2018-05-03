# Project Title

This code implements the algorithms described in the paper:
"Bridging Languages through Images with Deep Partial Canonical Correlation Analysis" by Guy Rotman, Ivan Vulić and Roi Reichart.\
Please cite the paper if you are using this code.

## Prerequisites
The code was implemented in python 3.6.3 with anaconda environment.
All requirements are included in the requirements.txt file.
They can be installed by running the following command from the command line:
`pip install -r requirements.txt`


## Data

.h5 files with all samples of the WIW dataset (including textual and visual features) are available in the following link:
[WIW Feature Set](https://mega.nz/#!eV0STDTR!w6Xg248RQdzL28VOmoqsFLidJqrlSZKx7f8AGqfA204).

Please unzip the wiw_data.zip file into the /data directory.

In order to run the models please make sure to first split the WIW dataset to train/val/test by running the following command from the command line:
`python split_wiw.py`

###### Optional
The full dataset (including the set of images) can be downloaded from the following link:
[WIW Dataset](https://mega.nz/#!Gc0kHBTA!CYpHo_Vs2j1BIML2rlBhxFtOzzAzpkhSIeYT3rE93Go)

## Models

The directory contains the following models:
1. dpcca_a.py - An implementation for Deep Partial Canonical Correlation Analysis by the NOI optimization algorithm for variant A.
2. dpcca_b.py - An implementation for Deep Partial Canonical Correlation Analysis by the NOI optimization algorithm for variant B.

- Each model can be executed (training + evaluation) by running the following command from the command line:
  `python model_name.py` (e.g. `python dpcca_a.py`).

- The architectures of the models are implemented in the model_architecture.py file.


## Hyperparameters and Default Settings

All hyperparameters and default settings appear in the cfg.py file.
A detailed explanation of them appears inside the file.

## Task and Evaluation

### Cross-Lingual Word Retrieval (Also known as Bilingual Lexicon Induction) 

The cross-lingual word retrieval task can be described as follows: Given a word in one language the goal is to retrieve the correct translation of it from a lexicon of a second language.

- To train and test on the EN-DE version of WIW please set in the cfg.py file:
```python
 self.feats = ['eng','ger','vis']
```
- To train and test on the EN-IT version of WIW please set in the cfg.py file:
```python 
self.feats = ['eng','it','vis']
```
- To train and test on the EN-RU version of WIW please set in the cfg.py file:
```python
 self.feats = ['eng','ru','vis']
```
#### Evaluation
Evaluation of R@K (Recall at K) for the task is implemented in the retrieval_eval.py file.

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details

## References

- English vectors were taken from: "Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
- German vectors were taken from: Ivan Vulic and Anna Korhonen. 2016a. Is "universal ´syntax" universally useful for learning distributed representations?"
In Proceedings of ACL, pages 518–524.
- Italian vectors were taken from: "Georgiana Dinu, Angeliki Lazaridou, and Marco Baroni. 2015. Improving zero-shot learning by mitigating the hubness problem. In Proceedings of ICLR: Workshop Papers."
- Russian vectors were taken from: "Andrey Kutuzov and Igor Andreev. 2015. Texts in, meaning out: neural language models in semantic similarity task for Russian. In Proceedings of DIALOG."
- Visual vectors were taken from: "Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014)."

