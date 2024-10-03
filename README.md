# Interpreting GPT-2 Predictions Using SHAP

*Using SHAP to generate local explanations for individual predictions from a pre-trained black-box model GPT-2.*

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
  - [Analyzing Racial Bias in GPT-2](#analyzing-racial-bias-in-gpt-2)
  - [Comparative Analysis](#comparative-analysis)
  - [Interpreting Predefined Outputs](#interpreting-predefined-outputs)
- [Why Choose SHAP Over LIME and Anchors](#why-choose-shap-over-lime-and-anchors)
- [Strengths, Limitations, and Potential Improvements of SHAP](#strengths-limitations-and-potential-improvements-of-shap)
- [References](#references)

---

## Introduction

This project demonstrates how to use SHAP (SHapley Additive exPlanations) to generate local explanations for individual predictions made by GPT-2, a pre-trained language model developed by OpenAI. By leveraging SHAP, we aim to interpret the influence of each input token on the model's output, providing insights into the model's decision-making process and uncovering potential biases.

---

## Prerequisites

- Python 3.6+
- Jupyter Notebook or Google Colab
- Basic understanding of machine learning and natural language processing
- Familiarity with GPT-2 and interpretability techniques

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ritu1412/SHAP-Interpretabilty.git
   ```

2. **Navigate to the Notebook Directory**

   ```bash
   cd SHAP-Interpretabilty/notebooks
   ```

3. **Install Required Packages**

   Install the necessary Python packages using `pip`:

   ```bash
   pip install transformers shap
   ```

---

## Usage Examples

### Import Libraries and Load GPT-2 Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shap

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.is_decoder = True
```

### Analyzing Racial Bias in GPT-2

#### **Input Sentence 1**

```python
sentence_1 = ["An African American teenager was walking down the street when"]
explainer = shap.Explainer(model, tokenizer)
shap_values_1 = explainer(sentence_1)
```

- **Model Output:**

  ```
  "he was shot in the head by a white police"
  ```

- **Visualization:**

  ```python
  shap.plots.text(shap_values_1)
  ```

  ![SHAP Values Sentence 1](images/shap_sentence1.png)

#### **Input Sentence 2**

```python
sentence_2 = ["An American teenager was walking down the street when"]
shap_values_2 = explainer(sentence_2)
```

- **Model Output:**

  ```
  "he was struck by a car. The driver"
  ```

- **Visualization:**

  ```python
  shap.plots.text(shap_values_2)
  ```

  ![SHAP Values Sentence 2](images/shap_sentence2.png)

### Comparative Analysis

By comparing the SHAP values of both sentences, we observe that the token **"African"** in Sentence 1 contributes significantly to a violent and racially charged continuation, whereas its absence in Sentence 2 leads to a neutral continuation. This highlights potential ethical biases in GPT-2's training data.

### Interpreting Predefined Outputs

#### **Input Sentences**

```python
x = [
    "I know quite a few people from India.",
    "I know quite a few people from America.",
    "I know quite a few people from Europe",
    "I know quite a few people from Spain"
]

y = [
    "They love their spicy food!",
    "They love their spicy food!",
    "They love their spicy food!",
    "They love their spicy food!"
]
```

#### **SHAP Analysis**

```python
teacher_forcing_model = shap.models.TeacherForcing(model, tokenizer)
masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)
explainer = shap.Explainer(teacher_forcing_model, masker)
shap_values = explainer(x, y)
```

#### **Visualization**

```python
shap.plots.text(shap_values)
```

![SHAP Values Predefined Outputs](images/shap_predefined.png)

#### **Interpretation**

- Tokens like **"India"** have positive SHAP values, indicating a strong association with **"spicy food"**.
- Tokens like **"Europe"** and **"Spain"** have negative SHAP values, showing less association.

---

## Why Choose SHAP Over LIME and Anchors

**SHAP** was chosen for this project due to its ability to provide consistent and fair attributions of each token's contribution to the model's output using Shapley values. Unlike **LIME**, which may not capture complex interactions effectively, and **Anchors**, which are better suited for classification tasks, SHAP:

- Handles feature dependencies and interactions crucial for language models.
- Provides granular insights at the token level.
- Offers robust visualization tools for better interpretability.

---

## Strengths, Limitations, and Potential Improvements of SHAP

### Strengths

- **Fair Attribution**
  - Provides consistent and fair attribution of each token's contribution using Shapley values.
- **Captures Complex Interactions**
  - Effectively handles feature dependencies and interactions in language models.
- **Model-Agnostic**
  - Applicable to any model without requiring internal modifications.

### Limitations

- **Computationally Intensive**
  - Resource-intensive for large models like GPT-2.
- **Approximation Necessity**
  - May require approximations reducing precision.
- **Interpretation Complexity**
  - Explanations can be complex and require expertise.

### Potential Improvements

- **Efficiency Enhancements**
  - Utilize sampling methods or approximate algorithms to reduce computation time.
- **Improved Visualizations**
  - Develop intuitive visualization tools tailored for NLP tasks.
- **Hybrid Approaches**
  - Combine SHAP with other interpretability techniques to enhance explanatory power.

---

## References

1. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems. [Link](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

2. **Molnar, C. (2019).** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. [Online Book](https://christophm.github.io/interpretable-ml-book/)

3. **SHAP Documentation.** [Link](https://shap.readthedocs.io/en/latest/)

4. **Transformers Documentation (Hugging Face).** [Link](https://huggingface.co/docs/transformers/index)

---

**Author:** Ritu Toshniwal

---

*Feel free to contribute to this project by opening issues or pull requests.*