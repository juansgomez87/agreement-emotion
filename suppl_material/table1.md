## Table 1
Song selection with emotion and quadrant information for annotation analysis, including synonyms for query and lyrics language. * refers to the annotation in the 4Q emotion data set. ** refers to audio features retrieved from the Spotify API: beats per minute (BPM), energy (A), and valence (V). Values in \textit{italics} show disagreement between the original annotations and Spotify audio features. We normalized the range for energy and valence to [-1,1] for comparison purposes.

| **Quadrant**  | **Emotion**       | **Synonym** | **Lang.** | **Artist – Song**                       | **BPM**** | **A  [-1, 1]**** | **V [-1, 1]**** |
|---------------|-------------------|-------------|-----------|-----------------------------------------|-----------|------------------|-----------------|
| Q1 \(A\+V\+\) | Joyful activation | joy         | Eng\.     | Taio Cruz \- Dynamite                   | 119\.98   | 0\.57            | 0\.63           |
|               |                   |             | Eng\.     | Miami Sound Machine \- Conga            | 122\.24   | 0\.31            | 0\.73           |
|               | Power             | \-          | Eng\.     | Ultra Montanes \- Anyway                | \-        | \-               | \-              |
|               |                   |             | Eng\.     | Rose Tattoo \- Rock n Roll Outlaw       | 93\.06    | 0\.54            | 0\.23           |
|               | Surprise          | \-          | Eng\.     | The Jordanaires \- Hound Dog            | 180\.67   | 0\.33            | 0\.04           |
|               |                   |             | Eng\.     | Shakira \- Animal City                  | 140\.03   | 0\.65            | 0\.78           |
| Q2 \(A\+V\-\) | Anger             | angry       | Eng\.     | Disincarnate \- In Sufferance           | 86\.79    | 0\.90            | \-0\.74         |
|               |                   |             | Inst\.    | Obituary \- Redneck Stomp               | 91\.06    | 0\.72            | \-0\.03         |
|               | Fear              | anguished   | Inst\.    | Joe Henry \- Nico Lost One Small Buddha | 98\.72    | \*0\.33\*        | \*0\.26\*       |
|               |                   |             | Eng\.     | Silverstein \- Worlds Apart             | 129\.91   | 0\.46            | \-0\.06         |
|               | Tension           | Tense       | Eng\.     | Pennywise \- Pennywise                  | 94\.28    | 0\.97            | 0\.31           |
|               |                   |             | Eng\.     | Squeeze \- Here Comes That Feeling      | 134\.03   | 0\.46            | \-0\.42         |
| Q3 \(A\-V\-\) | Bitterness        | bitter      | Eng\.     | Liz Phair \- Divorce Song               | 120\.37   | \*0\.69\*        | 0\.42\*         |
|               |                   |             | Eng\.     | Lou Reed \- Heroine                     | 76\.56    | \-0\.53          | \-0\.65         |
|               | Sadness           | sad         | Eng\.     | Motorhead \- Dead and Gone              | 102\.64   | 0\.43\*          | 0\.22\*         |
|               |                   |             | Spa\.     | Juan Luis Guerra \- Sobremesa           | 82\.36    | \-0\.46          | \-0\.70         |
| Q4 \(A\-V\+\) | Peace             | \-          | Eng\.     | Jim Brickman \- Simple Things           | 91\.99    | 0\.26\*          | 0\.29\*         |
|               |                   |             | Spa\.     | Gloria Estefan \- Mi Buen Amor          | 119\.94   | \-0\.22          | 0\.19           |
|               | Tenderness        | gentle      | Eng\.     | Celine Dion \- Beautiful Boy            | 111\.05   | \-0\.10          | 0\.21           |
|               |                   |             | Spa\.     | Beyonce \- Amor Gitano                  | 167\.88   | \*0\.49\*        | \*0\.19\*       |
|               | Transcendence     | spiritual   | Eng\.     | Steven Chapman \- Made for Worshipping  | 102\.89   | \*0\.55\*        | \*\-0\.56\*     |
|               |                   |             | Eng\.     | Matisyahu \- On Nature                  | 97\.01    | \*0\.64\*        | \*0\.30\*       |

## Table 2
Results from Median Test and Kruskal-Wallis H Test: German (N=374), English (N=572), Mandarin (N=594), Spanish (N=1232). Significance at p\< \0\.01 is depicted with \*\*.

| ****                   | ****          | **anger**   | **bitter**  | **fear**   | **joy** | **peace** | **power**   | **sad**    | **surprise** | **tender**  | **tension** | **transc.** |
|------------------------|---------------|-------------|-------------|------------|---------|-----------|-------------|------------|--------------|-------------|-------------|-------------|
| Median Test            | Median        | 2\.00       | 2\.00       | 1\.50      | 3\.00   | 2\.00     | 3\.00       | 2\.00      | 2\.00        | 2\.00       | 2\.00       | 2\.00       |
|                        | Chi\-Square   | 63\.000     | 20\.789     | 12\.636    | 5\.358  | 4\.010    | 100\.654    | 13\.474    | 13\.302      | 16\.071     | 45\.725     | 11\.531     |
|                        | Asymp\. Sig\. | 0\.0005\*\* | 0\.0005\*\* | 0\.005\*\* | 0\.147  | 0\.260    | 0\.0005\*\* | 0\.004\*\* | 0\.004\*\*   | 0\.001\*\*  | 0\.0005\*\* | 0\.009\*\*  |
| Kruskal\-Wallis H Test | Chi\-Square   | 46\.139     | 13\.107     | 12\.396    | 6\.063  | 7\.084    | 152\.318    | 12\.193    | 10\.414      | 21\.627     | 31\.703     | 20\.922     |
|                        | Asymp\. Sig\. | 0\.0005\*\* | 0\.004\*\*  | 0\.006\*\* | 0\.109  | 0\.069    | 0\.0005\*\* | 0\.007\*\* | 0\.015\*\*   | 0\.0005\*\* | 0\.0005\*\* | 0\.0005\*\* |

## Table 3




