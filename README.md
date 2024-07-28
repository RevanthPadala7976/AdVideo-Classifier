# AdVideo-Classifier
Answering 21 Binary questions with 150 Ad Videos

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-01453A?style=for-the-badge&logo=matplotlib&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-0078D4?style=for-the-badge&logo=machine-learning&logoColor=white)
![Natural Language Processing](https://img.shields.io/badge/Natural_Language_Processing-008080?style=for-the-badge&logo=natural-language-processing&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-FF5733?style=for-the-badge&logo=deep-learning&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-0096FF?style=for-the-badge&logo=bert&logoColor=white)



## Introduction:
The objective of this project is to utilize video advertisements and their corresponding text descriptions and speech captions to answer 21 binary (yes/no) questions. We explored multiple machine learning and deep learning approaches, including Logistic Regression, SVC, Random Forest, ANN, RNN, and BERT models. The goal was to maximize the performance metrics (precision, recall, F1 score, and accuracy) and compare the results across different models. Below is a detailed report of our approaches and their performance.

## Data Preparation
1. Data Reduction: The original ground truth dataset contained 449 rows. We reduced this dataset to 150 rows by aggregating responses for each `creative_data_id, ` selecting the most frequent response for each question. (agreement percentage among human coders)
2. Text Preprocessing: The transcriptions were preprocessed to remove stop words and convert text into numerical representations using TF-IDF vectorization for traditional models and tokenization for deep learning models.

## Traditional Classification Models
### Approach: 
Used TF-IDF vectorization followed by logistic Regression for each of the 21 binary questions. Each question was then treated as a separate binary classification problem. Logistic Regression, SVC, Random Forest, Gaussian NB was used to train a model for each question individually. Generated a classification report for each model and calculated the overall accuracy by taking the average accuracies of all questions.

**Accuracies:**
- Logistic Regression: 70.8%
- SVC: 73.1%
- Random Forest: 75.1%
- GaussianNB: 70.8%
  
## Neural Network Approach (ANN and RNN):
I built an ANN with multiple dense layers to predict multi label classification and got accuracy of 70%. RNN was implemented with LSTM layers to capture the sequential nature of text data. The text data was vectorized using TF-IDF, and achieved overall accuracy of 70.4%

### BERT Model Approach:
I used the pre-trained BERT model from Hugging Face for its superior capability in handling natural language. The model was fine-tuned for each of the 21 binary questions independently. Text data was tokenized using BERT’s tokenizer and fed into the model. The model achieved a precision of 0.1255, recall of 0.0871, and F1 Score of 0.0598. This performance was lower than expected, likely due to limited fine-tuning and dataset.

Random Forest was the best-performing model with an accuracy of 75.1%. Traditional models generally outperformed deep learning models, likely due to dataset size and training constraints. BERT's performance was lower than expected, suggesting the need for more extensive fine-tuning.

<img width="578" alt="accuracies" src="https://github.com/user-attachments/assets/c9e7d098-3128-412f-bb16-485af820fd0c">

## Further Analysis on Random Forest classifier: Individual question metrics

![Evaluation](https://github.com/user-attachments/assets/08e7d8b7-e168-47e1-95d3-dcdc24ff1835)

  
- Observation: Question 17 showed significantly lower performance.
- Human Coders Agreement: Agreement percentage among human coders was analyzed to understand inconsistencies.
- Prediction Agreement: Agreement percentage of model predictions was also calculated and compared with human coders.

**Insights**
- Potential Issues: Question 17's complexity or ambiguity could have contributed to lower agreement and performance.
- Recommendation: Further investigation into the question's phrasing and dataset quality is recommended to improve model performance.
  
## Conclusion:
In conclusion, the Random Forest model outperformed other models with an accuracy of 75.1% in classifying video advertisements into 21 binary questions. While traditional machine learning models, including Logistic Regression, SVC, and Gaussian Naive Bayes, demonstrated strong performance, deep learning models like ANN and RNN showed slightly lower accuracy. The BERT model, despite its advanced capabilities, underperformed, indicating a need for further fine- tuning and larger datasets. Future work will focus on optimizing BERT, exploring additional features, and potentially combining models to improve overall performance.
