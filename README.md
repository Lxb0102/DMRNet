# DMRNet

**This is the data and code for our paper** `Debiased Medication Recommendation through Fusing Frequent Pattern and Temporal Medical Records`.

For reproduction of medication prediction results in our paper, see instructions below.

## Overview

We have modularized and encapsulated the code into a more readable form. In brief, DMRNet can be divided into two parts: encoder and decoder, encoder mainly gengerates the representation of patients and decoder calculates the usage probability for each drug labels.

## Requirements

Make sure your local environment has the following installed:

* `pytorch>=1.12.1 & <=1.9`
* `numpy == 1.15.1`
* `python >= 3.8`
* `scikit-learn>=0.24.2`

#### Datasets

We provide the dataset in the [data](data/) folder.

| Data      | Source                                                   | Description                                                  |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| MIMIC-III | [This link](https://physionet.org/content/mimiciii/1.4/) | MIMIC-III is freely-available database from 2001 to 2012, which is associated with over forty thousand patients who stayed in critical care units |
| MIMIC-IV  | [This link](https://physionet.org/content/mimiciv/2.2/)  | MIMIC-IV is freely-available database between 2008 - 2019, which is associated with 299,712 patients who stayed in critical care units |

## Documentation

```
--src
  │--README.md
  │--data_loader.py
  │--train.py
  │--model_net.py
  │--outer_models.py
  │--util.py
  
--data
  │--ddi_A_final.pkl
  │--records_final_iii.pkl
  │--records_final_iv.pkl
  │--voc_final_iii.pkl
  │--voc_final_iv.pkl
  │--pattern_records_final.pkl
  │--pattern_records_final.pkl
  │--basic_combos_iii.pkl
  │--basic_combos_iv.pkl
  │--processing.py
```

## How to DMRNet 

### 1 Install IDE 

Our project is built on PyCharm Community Edition ([click here to get](https://www.jetbrains.com/products/compare/?product=pycharm-ce&product=pycharm)).

### 2 Environment setting

#### 2.1 Inpterpreter 

We recommend using `Python 3.11` or higher as the script interpreter. [Click here to get](https://www.python.org/downloads/release/python-3110/) `Python 3.11`. 

#### 2.2 Packages

First, install the [conda](https://www.anaconda.com/)
Then, create the conda environment through yaml file:

```
conda env create -f env.yaml
```

### 3 Start training

Please follow the steps below:

3.1 prepare data and process

  In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows:

  - download MIMIC data and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
  - download DDI data and put it in ./data/
  - run code ./data/processing.py

  Data information in ./data:

  - **records_final.pkl** is the input data with four dimension (patient_idx, visit_idx, medical model, medical id) where medical model equals 3 made of diagnosis, procedure and drug.
  - **voc_final.pkl** is the vocabulary list to transform medical word to corresponding idx.
  - **ddi_A_final.pkl** and are drug-drug adjacency matrix constructed from EHR and DDI dataset.
  - **pattern_records_final.pkl** is the medical records whose drugs are converted to combinations by FP Tree algorithm script.
  - **basic_combos.pkl** is the list of combinations generated by the FP Tree algorithm script.

  3.2 run train.py


## Performance of DMRNet 

Compared with existing methods, DMRNet  shows a significant advantage on several common metrics:


| Methods  | Jaccard             | PRAUC               | F1                  | DDI                 | AVG_MED           |
| -------- | ------------------- | ------------------- | ------------------- | ------------------- | ----------------- |
| LEAP     | 0.4521 / 0.4287     | 0.6549 / 0.5506     | 0.6138 / 0.5820     | 0.0731 / 0.0592     | 18.7138 / 11.5198 |
| RETAIN   | 0.4887 / 0.4239     | 0.7556 / 0.6798     | 0.6481 / 0.5791     | 0.0835 / 0.0939     | 20.4051 / 10.8602 |
| GAMENet  | 0.5237 / 0.4963     | 0.7775 / 0.7508     | 0.6783 / 0.6514     | 0.0861 / 0.0890     | 27.2145 / 18.4426 |
| MICRON   | 0.5260 / 0.4921     | 0.7773 / 0.7514     | 0.6790 / 0.6482     | 0.0641 / 0.0574     | 18.9800 / 16.8439 |
| SafeDrug | 0.5233 / 0.5000     | 0.7742 / 0.7485     | 0.6764 / 0.6557     | 0.0615 / 0.0575     | 19.9178 / 14.4705 |
| COGNet   | 0.5289 / 0.5081     | 0.7675 / 0.7547     | 0.6829 / 0.6627     | 0.0831 / 0.0837     | 28.0900 / 24.6800 |
| DGCL     | 0.5255 / 0.4993     | 0.7738 / 0.7535     | 0.6801 / 0.6542     | 0.0836 / 0.0735     | 28.6253 / 16.6284 |
| MoleRec  | 0.5271 / 0.4930     | 0.7717 / 0.7426     | 0.6816 / 0.6503     | 0.0726 / 0.0961     | 21.6489 / 18.7085 |
| RAREMed  | 0.5373 / 0.5092     | 0.7963 / 0.7683     | 0.6899 / 0.6629     | 0.0780 / 0.0750     | 17.4936 / 15.3435 |
| DMRNet   | **0.5514 / 0.5263** | **0.7981 / 0.7776** | **0.7027 / 0.6796** | **0.0631 / 0.0655** | 19.5486 / 17.0217 |


## Acknowledgement

We sincerely thank these repositories GAMENet (https://github.com/sjy1203/GAMENet), SafeDrug (https://github.com/ycq091044/SafeDrug), and COGNet(https://github.com/BarryRun/COGNet) for their well-implemented pipeline upon which we build our codebase.

## TODO

To make the experiments more efficient, we developed some experimental scripts, which will be released along with the paper later.
