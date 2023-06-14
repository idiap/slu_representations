# EFFECTIVENESS OF TEXT, ACOUSTIC, AND LATTICE-BASED REPRESENTATIONS IN SPOKEN LANGUAGE UNDERSTANDING TASKS

### Authors: 

#### *Esaú Villatoro-Tello, Srikanth Madikeri,Juan Zuluaga-Gomez, Bidisha Sharma, Seyyed Saeed Sarfjoo, Iuliia Nigmatulina, Petr Motlicek, Alexei V. Ivanov, Aravind Ganapathiraju*

###### Paper accepted at ICASSP 2023 Conference ([ICASSP'23 Proceedings](https://ieeexplore.ieee.org/document/10095168)) ([ARXIV Version](https://arxiv.org/abs/2212.08489))




## Abstract

We perform an exhaustive evaluation of different representations to address the intent classification problem in a Spoken Language Understanding (SLU) setup. We benchmark three types of systems to perform the SLU intent detection task: 1) text-based, 2) lattice-based, and a novel 3) multimodal approach. Our work provides a comprehensive analysis of what could be the achievable performance of different state-of-the-art SLU systems under different circumstances, e.g., automatically- *vs* manually-generated transcripts. We evaluate the systems on the publicly available SLURP spoken language resource corpus. Our results indicate that using richer forms of Automatic Speech Recognition (ASR) outputs, namely word-consensus-networks, allows the SLU system to improve in comparison to the 1-best setup (5.5% relative improvement). However, crossmodal approaches, i.e., learning from acoustic and text embeddings, obtains performance similar to the oracle setup, a relative improvement of 17.8% over the 1-best configuration, being a recommended alternative to overcome the limitations of working with automatically generated transcripts. This repository provides the source code used during our experimentation. 



![Overview of the considered NLU/SLU methodologies for our performed experiments.](/supplements/slu_experiments.png "Overview of the considered NLU/SLU methodologies for our performed experiments.")

----
## Requirements


----
## How to...

----
## Distributed Learning

This code was designed and prepared to work through Idiap's Sun's Grid Engine. Keep in mind that if you want to use this code in a single GPU, you'll need to make the proper modifications. 

----
## Citation

#### Plain

E. Villatoro-Tello et al., "Effectiveness of Text, Acoustic, and Lattice-Based Representations in Spoken Language Understanding Tasks," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10095168.


#### BIBTeX
    @INPROCEEDINGS{VillatoroEtAl_ICASSP_2023,
        author={Villatoro-Tello, Esaú and Madikeri, Srikanth and Zuluaga-Gomez, Juan and Sharma, Bidisha and Saeed Sarfjoo, Seyyed and Nigmatulina, Iuliia and Motlicek, Petr and Ivanov, Alexei V. and Ganapathiraju, Aravind},
        booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
        title={Effectiveness of Text, Acoustic, and Lattice-Based Representations in Spoken Language Understanding Tasks}, 
        year={2023},
        pages={1-5},
        doi={10.1109/ICASSP49357.2023.10095168}
    }
