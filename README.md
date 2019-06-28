# agreement-emotion

## Abstract

In our present study, we address the relation between the emotions perceived in pop and rock music and the language spoken by the listener. Two main research questions are addressed:

1. Are there differences/correlations between the emotions perceived in pop/rock music by
listeners raised with different mother tongues?
2. Do personal characteristics have an influence on the perceived emotions for listeners of a
given language?

Our hypothesis is that there will be higher agreement (as defined by the Krippendorff alpha
coefficient) in the perceived emotions by subjects that speak the same language.

We use fragments of pop and rock music since these musical styles can be considered as neutral and homogeneous even when sung in different languages. All fragments are in either English or Spanish. We use the emotion tags of the Geneva Emotion Music Scale (GEMS) and compliment with other emotion tags to rate the different fragments. To collect user ratings, we created online surveys in four languages (Spanish, English, German and Mandarin) using two excerpts per emotion, for a total of 22 excerpts. Besides emotion ratings, we also collect information about musical knowledge, musical taste, listeners’ familiarity with the stimuli, listeners’ understanding of the lyrics, and demographics. We aim to characterize perceived emotion
with respect to these factors, and attempt to replicate previous studies that show lower agreement in perceived emotions among subjects with more musical experience and knowledge.

## Usage


 python process_data.py [-h] -l LANGUAGE -c COMPLETE -r REMOVE -q QUADRANT
                        [-n NUMBER] [-f FILTER]

optional arguments:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Select language to process
  -c COMPLETE, --complete COMPLETE
                        Complete ratings [y] or drop missing/NaN ratings [n]
  -r REMOVE, --remove REMOVE
                        Keep neutral ratings [y] or not [n]
  -q QUADRANT, --quadrant QUADRANT
                        Process by quadrants [y] or by emotions [n]
  -n NUMBER, --number NUMBER
                        Number of surveys to process with random sampling
  -f FILTER, --filter FILTER
                        Select filter for data [preference, familiarity,
                        understanding]


[Agreement results](https://docs.google.com/spreadsheets/d/16rNh481Zs8CZTdJTmnME2i84R1JXFegIC0WcNZ5rds8/edit#gid=0)