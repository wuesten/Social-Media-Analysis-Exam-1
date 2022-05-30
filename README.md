<h1><center>«Portfolio-Exam Part I» </center></h1>
<h2><center>MADS-SMA </center></h2>
<h3><center>Author: Tom Wüsten </center></h3>

### Introduction
This paper deals about the evaluation from different approaches of sentiment analysis. The basis for this is a dataset of restaurant reviews from the yelp platform. The reviews are in English and rate restaurants in Hamburg. The rating is done by stars from 1-5, where 5 stars is the best rating. At the beginning, information about the data set will be gathered in the data exploration, which will be used for the pre-processing of the data. Based on this a Sentiment analysis will compare classic approaches like Bag of Words with newer approaches like Transformer. Furthermore, it will be discussed which methods can be used to train Transformer from the platform Huggingface on own data.

### Data Exploration
In the data analysis we want to get a picture of how the data is distributed. For this purpose, we want to look at the structure of the ratings and give an overview of the period in which the comments were created. <br>
Based on the first graph, we can see that the ratings of the restaurants are often good. The most votes have received 5 & 4 stars. It is remarkable that there are few bad ratings. In the second plot, we examine the text length of the comments. Most of the comments are in the range between 100 words. <br>
<img src="/output/distribution_text_rating.png" alt="Alt text" title="Optional title">

The plot below examines when comments have been written. The dataset includes ratings from 2006 to 2022, with the majority of ratings from 2015 & 2016. Furthermore, it can be seen that the months do not have a large influence on the visits. There is only a drop in ratings in November and December.<br>

<img src="/output/distribution_rating_year_month.png" alt="Alt text" title="Optional title">

### Fine-tuning of Transformer

In this section I want to demonstrate how to fine tune a pre-trained transformer. For this purpose, a pre-trained transformer was used as a starting point which was trained to classify reviews into a rating scale of 1-5. Fine-tuning is very resource-intensive, so you have to use batch processing. Huggingface provides a trainer that solves such things. In order to use the Huggingface trainer, the data must be put into the correct format. Huggingface provides a dataset library for this purpose. I wanna show the workflow from creating the correct dataset to fine tune the transformer , upload the fine-tuned transformer to huggingface hub and do prediction on the test data.

The first step is generate the data. Therefore we split the data into train, val and test data. The train and val data is used for fine tuning the transformer and the test data is used for the evaluation. The train and val data will be saved into a csv format, because the dataset libary can convert the data more easily when it's in a csv format.

```python
train_df, test_df = train_test_split(yelp_reviews, test_size=0.2)
train, val_df = train_test_split(train_df, test_size=0.2)
val_df = val_df[["stars", "text"]]
train = train[["stars", "text"]]
test_df = test_df[["stars", "text"]]
train.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
train.shape, val_df.shape, test_df.shape
```

In the next step the data is loaded from the two csv files.

```python
dataset = load_dataset("csv", data_files={"train": "data/train.csv", "test": "data/val.csv"})
```
The next step shows the tokenization of the train and val data. Additionally we drop columns and formatted into tensors.

```python
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("stars", "labels")
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
small_train_dataset = small_train_dataset.remove_columns(["token_type_ids"])
small_eval_dataset = small_eval_dataset.remove_columns(["token_type_ids"])
```

The TrainingArguments library from huggingface provides settings of the training. I'm using the default strategy and set only the batchsize to 8 and the epochs to 3 because my hardware can't handle more data.

```python
training_args = TrainingArguments(
    output_dir="./sentiment-analysis",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="steps",
)
```

In this step we define on which metrics the model should be improved.

```python
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

The trainer takes the model, the train and eval data, trainings parameter and the metric.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

Now we can easily fine tune the transformer.

```python
trainer.train()
```
After the training is done we can push the fine-tuned transformer to Huggingface hub. But before you can do this you need to registrate. My fined tuned transformer can be found here: [Fine-tuned Transformer](https://huggingface.co/wuesten/sentiment-analysis-fh-kiel)

```python
tokenizer.push_to_hub("sentiment-analysis-fh-kiel")
model.push_to_hub("sentiment-analysis-fh-kiel")
```
After training we saved the Transformer locally, since it takes up a lot of space and we have to give a repo, I also published it on Huggingface. Now you can just download the model.

```python
my_model_from_hub = AutoModelForSequenceClassification.from_pretrained(
    "wuesten/sentiment-analysis-fh-kiel"
)
my_tokenizer = AutoTokenizer.from_pretrained("wuesten/sentiment-analysis-fh-kiel")
```
In this step we create a pipeline which gets the fine tuned transformer and the tokenizer as input.

```python

sentiment_pipeline_2 = pipeline("sentiment-analysis", my_model_from_hub, tokenizer=my_tokenizer)
```
In the last step, we can use the pipeline to evaluate the test data and create an accuracy.

```python
test_df["amount_text"] = test_df["text"].apply(lambda x: sum(word in x for word in x))
test_df = test_df[test_df["amount_text"] <= 512]
test_df["rating"] = test_df["text"].apply(lambda x: sentiment_pipeline_2(x)[0].get("label"))
test_df["rating"] = test_df["rating"].apply(lambda x: int(x[0]))
test_df = test_df.astype({"stars": int})
acc_score = accuracy_score(test_df["stars"] + 1, test_df["rating"])
```
### Results

|                      model 	| accuracy 	|
|---------------------------:	|---------:	|
|                Transformer 	| 0.630378 	|
|             Transformer EN 	| 0.623523 	|
|     fine-tuned Transformer 	| 0.832487 	|
|   Bag of words: GaussianNB 	| 0.365289 	|
|  Bag of words: linear Reg. 	| 0.320661 	|
| Bag of words: RandomForest 	| 0.525620 	|
|          TF-ID: GaussianNB 	| 0.370248 	|
|         TF-ID: linear Reg. 	| 0.438017 	|
|        TF-ID: RandomForest 	| 0.519008 	|

<img src="/output/conf_overview.png" alt="Alt text" title="Optional title">
