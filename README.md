# JaQuAD: Japanese Question Answering Dataset

## Overview

Japanese Question Answering Dataset (JaQuAD), released in 2022, is a
human-annotated dataset created for Japanese Machine Reading Comprehension.
JaQuAD is developed to provide a SQuAD-like QA dataset in Japanese.
JaQuAD contains 39,696 question-answer pairs.
Questions and answers are manually curated by human annotators.
Contexts are collected from Japanese Wikipedia articles.

For more information on how the dataset was created, refer to our paper,
[JaQuAD: Japanese Question Answering Dataset for Machine Reading
Comprehension](https://arxiv.org/abs/2202.01764).

## Data

JaQuAD consists of three sets: `train`, `validation`, and `test`. They were
created from disjoint sets of Wikipedia articles. The following table shows
statistics for each set:

Set | Number of Articles | Number of Contexts | Number of Questions
--------------|--------------------|--------------------|--------------------
Train | 691 | 9713 | 31748
Validation | 101 | 1431 | 3939
Test | 109 | 1479 | 4009

You can also download our dataset [here](
https://huggingface.co/datasets/SkelterLabsInc/JaQuAD).
(The `test` set is not publicly released yet.)

```python
from datasets import load_dataset
jaquad_data = load_dataset('SkelterLabsInc/JaQuAD')
```


## Baseline

We also provide a baseline model for JaQuAD for comparison. We created this
model by fine-tuning a publicly available Japanese BERT model on JaQuAD. You can
see the performance of the baseline model in the table below.

For more information on the model's creation, refer to
[JaQuAD.ipynb](JaQuAD.ipynb).

Pre-trained LM | Dev F1 | Dev EM | Test F1 | Test EM
---------------|--------|--------|---------|---------
BERT-Japanese | 77.35 | 61.01 | 78.92 | 63.38

You can download the baseline model [here](
https://huggingface.co/SkelterLabsInc/bert-base-japanese-jaquad).

## Usage

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

question = 'アレクサンダー・グラハム・ベルは、どこで生まれたの?'
context = 'アレクサンダー・グラハム・ベルは、スコットランド生まれの科学者、発明家、工学者である。世界初の>実用的電話の発明で知られている。'

model = AutoModelForQuestionAnswering.from_pretrained(
    'SkelterLabsInc/bert-base-japanese-jaquad')
tokenizer = AutoTokenizer.from_pretrained(
    'SkelterLabsInc/bert-base-japanese-jaquad')

inputs = tokenizer(
    question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
outputs = model(**inputs)
answer_start_scores = outputs.start_logits
answer_end_scores = outputs.end_logits

# Get the most likely start of the answer with the argmax of the score.
answer_start = torch.argmax(answer_start_scores)
# Get the most likely end of the answer with the argmax of the score.
# 1 is added to `answer_end` because the index of the score is inclusive.
answer_end = torch.argmax(answer_end_scores) + 1

answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
# answer = 'スコットランド'
```

## Limitations

This dataset is not yet complete.
The social biases of this dataset have not yet been investigated.

If you find any errors in JaQuAD, please contact <jaquad@skelterlabs.com>.

## Reference

If you use our dataset or code, please cite our paper:

```bibtex
@misc{so2022jaquad,
      title={{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
      author={ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
      year={2022},
      eprint={2202.01764},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## LICENSE

The JaQuAD dataset is licensed under the [CC BY-SA 3.0]
(https://creativecommons.org/licenses/by-sa/3.0/) license.

## Have Questions?

Ask us at <jaquad@skelterlabs.com>.
