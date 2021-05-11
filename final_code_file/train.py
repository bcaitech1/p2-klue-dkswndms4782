from sklearn.metrics import accuracy_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from load_data import *
from sklearn.model_selection import train_test_split
from transformers import ElectraConfig,ElectraTokenizer, ElectraModel, ElectraForSequenceClassification
# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():
  # load model and tokenizer
  Model_name = "xlm-roberta-large"
  # Model_name = 'google/electra-base-discriminator'
  tokenizer = XLMRobertaTokenizer.from_pretrained(Model_name)

  # split dataset
  # dataset = pd.read_csv('/opt/ml/input/data/train/train_with_pororo.txt', delimiter='\t', header=None)
  # train, dev = train_test_split(dataset, test_size=0.2, random_state=42)
  # train.to_csv('/opt/ml/input/data/train/train_train.tsv', sep='\t', header=None, index=False)
  # dev.to_csv('/opt/ml/input/data/train/train_dev.tsv', sep='\t', header=None, index=False)

  # load dataset
  dataset = load_data('/opt/ml/input/data/train/train.tsv')
  # dev_dataset = load_data('/opt/ml/input/data/train/train_dev.tsv')

  label = dataset['label'].values
  # dev_label = dev_dataset['label'].values

  # tokenizing dataset
  tokenized_train = tokenized_dataset(dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, label)
  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  electra_config = XLMRobertaConfig.from_pretrained(Model_name)
  electra_config.num_labels = 42
  model = XLMRobertaForSequenceClassification.from_pretrained(Model_name, config=electra_config)
  model.to(device)

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results/bert',
    save_total_limit=3,
    save_steps=100,
    num_train_epochs=10,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    # per_device_eval_batch_size=32,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    # evaluation_strategy='steps',
    # eval_steps = 100,
    dataloader_num_workers=4,
    label_smoothing_factor=0.5
  )
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RE_train_dataset,
    # eval_dataset=RE_dev_dataset,
    # compute_metrics=compute_metrics
  )

  # train model
  trainer.train()

def main():
  train()

if __name__ == '__main__':
  main()
