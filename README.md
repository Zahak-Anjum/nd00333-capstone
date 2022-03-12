
# Survival Prediction With Heart Failure Clinical Record Dataset Using Machine Learning On Microsoft Azure


## Table of contents
   * [Overview](#Overview)
   * [Project Set Up and Installation](#Project-Set-Up-and-Installation)
   * [Dataset](#Dataset)
   * [Automated ML](#Automated-ML)
   * [Hyperparameter Tuning](#Hyperparameter-Tuning)
   * [Model Deployment](#Model-Deployment)
   * [Screen Recording](#Screen-Recording)
   * [Stand out suggestions](#Comments-and-standout-improvements)
   * [References](#References)

***

### Overview

The current project uses machine learning on Microsoft Azure to predict survival chance of patients based on their medical data which is available at https://www.kaggle.com/andrewmvd/heart-failure-clinical-data. 

I create two models in the environment of Azure Machine Learning Studio
1. Using Automated Machine Learning (i.e. AutoML)
2. Using Customized model whose hyperparameters are tuned using HyperDrive.

The two models are then compare the performance of both models and deploy the best performing model as a service using Azure Container Instances (ACI).
model workflow.PNG

![Project Workflow](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/workflow%20capstone.PNG)

## Project Set Up and Installation
In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks:

- `automl.ipynb`: for the AutoML experiment;
- `hyperparameter_tuning.ipynb`: for the HyperDrive experiment.

The following files are also required

- `heart_failure_clinical_records_dataset.csv`: the dataset file. It can also be taken directly from Kaggle; 
- `train.py`: a basic script for manipulating the data used in the HyperDrive experiment;
- `scoring.py`: the script used to deploy the model which is downloaded from within Azure Machine Learning Studio; &
- `env.yml`: the environment file which is also downloaded from within Azure Machine Learning Studio.

## Dataset

The dataset used is taken from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) and -as we can read in the original [Research article](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5)- the data comes from 299 patients with heart failure collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015. The patients consisted of 105 women and 194 men, and their ages range between 40 and 95 years old.

The dataset contains 13 features:

| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| *age* | Age of patient | Years (40-95) |
| *anaemia* | Decrease of red blood cells or hemoglobin | Boolean (0=No, 1=Yes) |
| *creatinine-phosphokinase* | Level of the CPK enzyme in the blood | mcg/L |
| *diabetes* | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes) |
| *ejection_fraction* | Percentage of blood leaving the heart at each contraction | Percentage |
| *high_blood_pressure* | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes) |
| *platelets* | Platelets in the blood | kiloplatelets/mL	|
| *serum_creatinine* | Level of creatinine in the blood | mg/dL |
| *serum_sodium* | Level of sodium in the blood | mEq/L |
| *sex* | Female (F) or Male (M) | Binary (0=F, 1=M) |
| *smoking* | Whether the patient smokes or not | Boolean (0=No, 1=Yes) |
| *time* | Follow-up period | Days |
| *DEATH_EVENT* | Whether the patient died during the follow-up period | Boolean (0=No, 1=Yes) |

### Task
The main task that I seek to solve with this project & dataset is to classify patients based on their odds of survival. The prediction is based on the first 12 features included in the above table, while the classification result is reflected in the last column named _Death event (target)_ and it is either 0 (No) or 1 (Yes).

### Access
First, I made the data publicly accessible in the my GitHub repository. It is availabel under the link: https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/heart_failure_clinical_records_dataset.csv

## Automated ML
The automl.ipynb file is present in the starter_file folder. Below are the training configurations and settings used for the model to train via "Automl" run in automl.ipynb file.

```
automl_settings = {"n_cross_validations": 2,
                   "primary_metric": 'accuracy',
                   "enable_early_stopping": True,
                   "max_concurrent_iterations": 4,
                   "experiment_timeout_minutes": 20,
                   "verbosity": logging.INFO
                  }
```

```
automl_config = AutoMLConfig(compute_target = compute_target,
                             task = 'classification',
                             training_data = dataset,
                             label_column_name = 'DEATH_EVENT',
                             path = project_folder,
                             featurization = 'auto',
                             debug_log = 'automl_errors.log,
                             enable_onnx_compatible_models = False
                             **automl_settings
                             )
```

After that experiment is being submitted using:

```
remote_run = experiment.submit(automl_config, show_output = True)
remote_run.wait_for_completion()
```
### Results
The Automl experiment ran for 20 minutes with 39 iterations. Below you can observe the child runs of the experiment as:



![child runs automl](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/run%20widgets.PNG)


The best model was observed with 'runId': 'AutoML_b3b7183d-b3ce-45c6-b7cb-75498698ab46_38' and "class_name":"Ensemble". Below is the screenshot of the model with the best run id and further details.

![Bestmodel](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/best%20automl%20run.PNG)

Also you can see in the screenshot below of Azure ML studio that the best model with 'runId': 'AutoML_b3b7183d-b3ce-45c6-b7cb-75498698ab46_38' of 'Accuracy' '0.8629. 


![azure bestmodelscreenshot](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/automl%20bestrun%20accuray.PNG)


## Hyperparameter Tuning
The hyperparameter_tuning file is present in the starter_file folder. For this experiment I used a custom Scikit-learn Logistic Regression model, whose hyperparameters I am optimising using HyperDrive. Logistic regression is best suited for binary classification models like this one and this is the main reason to run Logistic Regression.

I specify the parameter sampler using the parameters C and max_iter and chose discrete values with choice for both parameters.

**Parameter sampler**

I specified the parameter sampler as such:

```
ps = RandomParameterSampling(
    {
        '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
        '--max_iter': choice(50,100,200,300)
    }
)
```

I chose discrete values with _choice_ for both parameters, _C_ and _max_iter_.

_C_ is the Regularization while _max_iter_ is the maximum number of iterations.

_RandomParameterSampling_ is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. If budget is not an issue, we could use _GridParameterSampling_ to exhaustively search over the search space or _BayesianParameterSampling_ to explore the hyperparameter space. 

**Early stopping policy**

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the _BanditPolicy_ which I specified as follows:
```
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish and this is the reason I chose it.


### Results
The Hyperdrive experiment ran for 'max_total_runs=16'. Below you can see in the screenshot the child runs for Hyperdrive experiment.

![hyperdrivewidget](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/widgets%20hyperdrive.PNG)


The best model was with the 'runId': 'HD_79212944-9c6b-4259-bbf8-77869f74ed15_1' as you see from the screenshot below with the 'Accuracy' of '0.83'.

![hyperdrive best model](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/hyperdrive%20best%20model.PNG)

Also it can also be observed that the best model with best run ID and metric Accuracy throught the screenshot below from the AzureMLstudio. 

![hyperdrivebestrun azure](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/hyperdrive%20best%20run.PNG)

## Model Deployment
Using as basis the `accuracy` metric, we can state that the best AutoML model is superior to the best model that resulted from the HyperDrive run. For this reason, I choose to deploy the best model from AutoML run (`best_run_automl.pkl`, Version 1)

The deployment is done following the steps below:


* Inference configuration
* Entry script
* Choosing a compute target
* Deployment of the model
* Testing the resulting web service

### Inference configuration

The inference configuration defines the environment used to run the deployed model. The inference configuration includes two entities, which are used to run the model when it's deployed.


### Entry script

The entry script is the `scoring.py` file. The entry script loads the model when the deployed service starts and it is also responsible for receiving data, passing it to the model, and then returning a response.

### Compute target

As compute target, I chose the Azure Container Instances (ACI) service, which is used for low-scale CPU-based workloads that require less than 48 GB of RAM.

The ACI Webservice Class represents a machine learning model deployed as a web service endpoint on Azure Container Instances. The deployed service is created from the model, script, and associated files, as I explain above. The resulting web service is a load-balanced, HTTP endpoint with a REST API. We can send data to this API and receive the prediction returned by the model.

### Deployment

Bringing all of the above together, here is the actual deployment in action:

![deployementnotebook](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/endpoint%20automl%20notebook.PNG)

Deployment takes some time to conclude, but when it finishes successfully the ACI web service has a status of ***Healthy*** and the model is deployed correctly. Below is the screen shot of ML azure studio.

![azure endpoint healthy screenshot](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/azure%20endpoint%20automl.PNG)

* Testing the resulting web service

To consume the endpoints after the model is being deployed I have use a python ile name as `endpoint.py` and provided the key and scoring uri. To see if the I am getting post and get request from the model endpoint I have ran the endpoint.py while with the data file in the data folder. The response was recorded and the model was performing as expected. Here is the screen shot of consuming endpoints results.

![endpoint consume](https://github.com/Zahak-Anjum/nd00333-capstone/blob/master/endpoint%20result.PNG)


## Screen Recording
The screen recording can be found [here](https://drive.google.com/file/d/16QjaO7nzdnekWl6z941BdYtdbdQvcG4U/view) and it shows the project in demonstration which include:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
* The first factor that could improve the model is increasing the training time. It would be great to be able to experiment more with the hyperparameters chosen for the HyperDrive model or even try running it with more of the available hyperparameters, with less time contraints.

* In order to test the deployed service, another way would be using the Swagger URI of the deployed service and the [Swagger UI](https://swagger.io/tools/swagger-ui/).

* I would definetly like to explore different models and configuration to explore the best and relevent results.  

* Last but not the least, the question of how much training data is required for machine learning is always valid and, by all means, the dataset used here is rather small and  contains the medical records of only 299 patients. Increasing the sample size can mean higher level of accuracy and more reliable results.

## References

- Udacity Nanodegree material
- [Heart Failure Prediction Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)
- [Consume an Azure Machine Learning model deployed as a web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)
- [Deploy machine learning models to Azure](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli)
- [AutoMLConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py)
- [Using Azure Machine Learning for Hyperparameter Optimization](https://dev.to/azure/using-azure-machine-learning-for-hyperparameter-optimization-3kgj)
- [hyperdrive Package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py)
- [Tune hyperparameters for your model with Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
