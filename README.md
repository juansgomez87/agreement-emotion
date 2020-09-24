# agreement-emotion

## Abstract

Tagging a musical excerpt with an emotion label may result in a vague and ambivalent exercise. This subjectivity entangles several high-level music description tasks when the computational models built to address them produce predictions on the basis of a "ground truth". In this study, we investigate the relationship between emotions perceived in pop and rock music (mainly in Euro-American styles) and personal characteristics from the listener, using mother language as a key feature. Our goal is to understand the influence of lyrics comprehension on music emotion perception and use this knowledge to improve Music Emotion Recognition (MER) models. 

We systematically analyze over 30K annotations of 22 musical fragments to assess the impact of individual differences on agreement, as defined by Krippendorff's alpha coefficient. We employ personal characteristics to form group-based annotations by assembling ratings with respect to listeners' familiarity, preference, lyrics comprehension, and music sophistication. Finally, we study our group-based annotations in a two-fold approach: (1) assessing the similarity within annotations using manifold learning algorithms and unsupervised clustering, and (2) analyzing their performance by training classification models with diverse "ground truths". Our results suggest that a) applying a broader categorization of taxonomies and b) using multi-label, group-based annotations based on language, can be beneficial for MER models.

## Usage

### Agreement analysis

The process_data.py file allows to analyze annotations for agreement and cluster annotations using different manifold learning algorithms. For an one of the examples seen in the paper, run the following code:
```python
python3 process_data.py -l a -c n -r n -q n -lf lyrics -clu y -f u1
```
This script analyzes agreement across all surveys (-l a), uses only full responses (-c n), uses raw annotations (-r n), does not map emotions to quadrants (-q n), uses only songs with lyrics (-lf lyrics), processes clusters using manifold learning (-clu y), and filters data w.r.t. positive understanding of lyrics (-f u1).

For usage flags, use:
```python
python3 process_data.py -h
```

### Classification
The classifier.py file trains a Support Vector Machine classifier and compares the performance of different "ground truths". For one of the examples seen in the paper, run the following code:
```python
python3 classifier.py -l a -q n -m svm -nc 8 -f u1
```
This script trains classifiers using annotations from all surveys (-l a), does not map emotions to quadrants (-q n), uses a support vector machine classifier (-m svm), uses 8 components for PCA (-nc 8), and filters data w.r.t. positive understanding of lyrics (-f u1).

For usage flags, use:
```python
python3 classifier.py -h
```

## Publication
```
@InProceedings{GomezCanon2020ISMIR,
    author = {Juan Sebasti{\'a}n G{\'o}mez-Ca{\~n}{\'o}n and Estefan{\'i}a Cano and Perfecto Herrera and Emilia G{\'o}mez},
    title = {Joyful for you and tender for us: the influence of individual characteristics and language on emotion labeling and classification},
    booktitle = {Proceedings of the 21th International Society for Music Information Retrieval Conference (ISMIR)},
year = {2020},
    location = {Montr{\'e}al, Canada},
    pages = {},
}
```
