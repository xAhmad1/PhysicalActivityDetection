# **Physical Activity Prediciton**

### Team Members:

 - **Abed El Kader EL SHAAR**
 - **Ahmad AL MASRI**
 - **Ahmad KHALIFE**
 - **Hadi JABER**
 - **Karen KHOURY**
 - **Sara GHAMLOUSH**

This challenge was done as a project for the Master 2 Data Science (2022/2023), DATACAMP course
## Introduction
Physical activity plays an important role in controlling obesity and maintaining a healthy living. It becomes increasingly important during a pandemic due to restrictions on outdoor activities.
Tracking physical activities using miniature wearable sensors and state-of-the-art machine learning
techniques can encourage healthy living and control obesity. 

This project focuses on introducing novel techniques to identify and log physical activities using machine learning techniques and wearable sensors. Physical activities performed in daily life are often unstructured and unplanned, and one
activity or set of activities (sitting, standing) might be more frequent than others (walking, stairs
up, stairs down).

The outcome of this project should be to be able to classify activities based on the different sensors readings, and thus gain insights on what muscles are used to do different activities and what activities might be easier or more common than others.
## Describtion of the data
The dataset used is [PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)(Physical Activity Monitoring for Aging People). It was created in autumn 2011 for the purpose of monitoring physical activity in older individuals. It includes data from 9 subjects, wearing 3 IMUs and a HR-monitor, and performing 18 different activities. Over 10 hours of data were collected altogether, from which nearly 8 hours were labeled as one of the 18 activities.

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.



### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)